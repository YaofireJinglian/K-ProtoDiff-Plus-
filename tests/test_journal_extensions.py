import unittest

import torch

from Models.KPD.k_diffusion import FDM, Model


class JournalExtensionTests(unittest.TestCase):
    def test_frequency_bands_reconstruct_real_input(self):
        x = torch.randn(3, 24, 5)
        low, mid, high = FDM(fl_ratio=0.1, fh_ratio=0.3)(x)
        self.assertTrue(torch.allclose(low + mid + high, x, atol=1e-5, rtol=1e-5))

    def test_multiscale_prototypes_receive_gradients(self):
        model = self._small_model()
        x = torch.randn(2, 24, 5)
        loss = model(x)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn('prototype_reconstruction', model._last_loss_components)
        for prototypes in model.kpa.prototypes:
            self.assertIsNotNone(prototypes.grad)
            self.assertTrue(torch.isfinite(prototypes.grad).all())

    def test_conference_defaults_remain_legacy(self):
        model = Model(
            seq_length=24,
            feature_size=5,
            d_model=16,
            timesteps=8,
            sampling_timesteps=8,
            n_heads=4,
        )
        self.assertEqual(model.prototype_mode, 'point')
        self.assertEqual(model.sampling_mode, 'legacy')
        self.assertEqual(model.fdm.implementation, 'legacy')

    def test_adaptive_sampler_respects_nfe_budget(self):
        model = self._small_model(adaptive_sampling_timesteps=4)
        calibration_batch = torch.randn(4, 24, 5)
        model.calibrate_reflection(calibration_batch)
        model.eval()
        samples = model.generate_mts(batch_size=2)

        self.assertEqual(tuple(samples.shape), (2, 24, 5))
        self.assertLessEqual(model.last_sampling_stats['nfe'], 4)
        self.assertEqual(
            model.last_sampling_stats['nfe'],
            len(model.last_sampling_stats['timesteps']),
        )
        self.assertIn('mean_roughness_ratio', model.last_sampling_stats)

    @staticmethod
    def _small_model(**overrides):
        params = dict(
            seq_length=24,
            feature_size=5,
            n_layer_enc=1,
            n_layer_dec=1,
            d_model=16,
            timesteps=8,
            sampling_timesteps=8,
            n_heads=4,
            prototype_mode='multiscale',
            prototype_scales=[4, 8, 12],
            sampling_mode='adaptive_reflection',
            adaptive_sampling_timesteps=4,
            use_ff=False,
        )
        params.update(overrides)
        return Model(**params)


if __name__ == '__main__':
    unittest.main()
