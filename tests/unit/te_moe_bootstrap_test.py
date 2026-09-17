# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests of EP bootstrap sizing; no TE installation or GPUs required."""

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from maxtext.utils import max_utils


class TeMoeBootstrapTest(unittest.TestCase):
  """Bootstrap must reserve a per-call batch, not the entire GA batch."""

  def setUp(self):
    super().setUp()
    self.config = SimpleNamespace(
        te_moe_block=True,
        gradient_accumulation_steps=3,
        micro_batch_size_to_train_on=24,
        micro_batch_size_to_eval_on=24,
        eval_interval=-1,
        num_experts=16,
        num_experts_per_tok=8,
        ragged_buffer_factor=2.0,
        moe_expert_input_dim=0,
        emb_dim=7168,
        dtype=jnp.bfloat16,
    )
    self.mesh = mock.MagicMock(shape={"fsdp": 2, "expert": 2})
    ep_module = ModuleType("transformer_engine.jax.ep")
    moe_module = ModuleType("transformer_engine.jax.moe")
    self.bootstrap = ep_module.ep_bootstrap = mock.Mock()
    self.record = moe_module.record_ep_bootstrap_signature_for_moe = mock.Mock()
    # Treat TE's capacity calculation as opaque. Check its input token bound
    # and that the resulting capacity is passed unchanged to bootstrap/record.
    self.capacity = moe_module.get_moe_recv_capacity_per_rank = mock.Mock(return_value=393216)
    self.enterContext(mock.patch.dict(sys.modules, {ep_module.__name__: ep_module, moe_module.__name__: moe_module}))
    self.enterContext(mock.patch.object(max_utils, "_te_moe_bootstrap_signature", None))
    self.enterContext(mock.patch.object(jax, "local_device_count", return_value=1))
    self.enterContext(mock.patch.object(jax, "process_count", return_value=4))
    self.enterContext(mock.patch.object(jax, "process_index", return_value=0))
    self.enterContext(mock.patch.object(jax, "set_mesh", return_value=nullcontext()))

  def bootstrap_batch(self, loaded_batch=72):
    shaped_batch = {"inputs": jax.ShapeDtypeStruct((loaded_batch, 4096), jnp.int32)}
    max_utils.maybe_bootstrap_te_moe(self.config, self.mesh, shaped_batch)

  def assert_token_bound(self, expected):
    self.assertEqual(self.bootstrap.call_args.kwargs["max_tokens_per_rank"], expected)
    self.assertEqual(self.record.call_args.kwargs["max_tokens_per_rank"], expected)
    self.assertEqual(self.capacity.call_count, 2)
    for call in self.capacity.call_args_list:
      self.assertEqual(call.kwargs["max_tokens_per_rank"], expected)

  def test_ga3_uses_one_microbatch(self):
    self.bootstrap_batch()
    self.assert_token_bound(24576)
    self.assertEqual(self.bootstrap.call_args.kwargs["recv_capacity_per_rank"], 393216)
    self.assertEqual(max_utils.get_te_moe_recv_capacity_per_rank(), 393216)
    self.assertEqual(self.record.call_args.kwargs["recv_capacity_per_rank"], 393216)
    self.assertEqual(self.capacity.call_args.kwargs["recv_capacity_factor"], 2.0)
    self.assertEqual(self.bootstrap.call_args.kwargs["hidden_dim"], 7168)
    self.assertEqual(self.bootstrap.call_args.kwargs["max_token_dtype"], jnp.bfloat16)

  def test_ga1_preserves_batch_size(self):
    self.config.gradient_accumulation_steps = 1
    self.bootstrap_batch(24)
    self.assert_token_bound(24576)

  def test_larger_enabled_eval_batch_is_covered(self):
    self.config.eval_interval = 10
    self.config.micro_batch_size_to_eval_on = 48
    self.bootstrap_batch()
    self.assert_token_bound(49152)

  def test_smaller_enabled_eval_does_not_shrink_train_bound(self):
    self.config.eval_interval = 10
    self.config.micro_batch_size_to_eval_on = 12
    self.bootstrap_batch()
    self.assert_token_bound(24576)

  def test_disabled_eval_does_not_inflate_bound(self):
    self.config.micro_batch_size_to_eval_on = 96
    self.bootstrap_batch()
    self.assert_token_bound(24576)

  def test_expanded_loader_keeps_conservative_microbatch_bound(self):
    self.bootstrap_batch(144)
    self.assert_token_bound(49152)

  def test_rampup_batch_covers_final_training_initialization(self):
    self.bootstrap_batch(36)
    self.assert_token_bound(24576)

  def test_invalid_ga_batch_raises_before_bootstrap(self):
    with self.assertRaisesRegex(ValueError, "divisible by positive gradient_accumulation_steps"):
      self.bootstrap_batch(73)
    self.bootstrap.assert_not_called()

  def test_nonpositive_ga_raises_before_bootstrap(self):
    for ga_steps in (0, -1):
      with self.subTest(ga_steps=ga_steps):
        self.config.gradient_accumulation_steps = ga_steps
        with self.assertRaisesRegex(ValueError, "positive gradient_accumulation_steps"):
          self.bootstrap_batch()
    self.bootstrap.assert_not_called()

  def test_unshardable_microbatch_raises_before_bootstrap(self):
    self.config.micro_batch_size_to_train_on = 25
    with self.assertRaisesRegex(ValueError, r"divisible by FSDP\*EP=4"):
      self.bootstrap_batch(75)
    self.bootstrap.assert_not_called()

  def test_same_shape_does_not_bootstrap_twice(self):
    self.bootstrap_batch()
    self.bootstrap_batch()
    self.bootstrap.assert_called_once()
    self.record.assert_called_once()

  def test_non_te_model_skips_bootstrap(self):
    self.config.te_moe_block = False
    self.bootstrap_batch()
    self.bootstrap.assert_not_called()
    self.capacity.assert_not_called()


if __name__ == "__main__":
  unittest.main()
