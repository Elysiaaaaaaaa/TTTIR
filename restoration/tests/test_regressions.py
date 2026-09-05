"""Run from restoration/: python -m unittest discover -s tests -v."""
import contextlib
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.lowlight_model import enhancement_model
from models.loss import SSIM
from test_derain import _eval


class RestorationRegressionTests(unittest.TestCase):
    def test_derain_metrics_compare_against_rgb_ground_truth(self):
        class FixedPrediction(torch.nn.Module):
            def forward(self, image):
                prediction = torch.full_like(image, 0.6)
                return [prediction, prediction, prediction]

        model = FixedPrediction()
        rgb_input = torch.zeros(1, 3, 32, 32)
        ycbcr_input = torch.full_like(rgb_input, 0.2)
        ground_truth = torch.full_like(rgb_input, 0.5)
        batch = (rgb_input, ycbcr_input, ground_truth, ['sample.png'])
        args = SimpleNamespace(test_model='unused.pkl', data_dir='unused', save_image=False)
        output = io.StringIO()

        with patch('test_derain.torch.load', return_value={'model': model.state_dict()}), \
                patch('test_derain.test_dataloader', return_value=[batch]), \
                patch('torch.cuda.is_available', return_value=False), \
                contextlib.redirect_stdout(output):
            _eval(model, args)

        self.assertIn('The average PSNR is 20.00 dB', output.getvalue())
        expected_ssim = (2 * 0.6 * 0.5 + 0.01 ** 2) / (0.6 ** 2 + 0.5 ** 2 + 0.01 ** 2)
        reported = output.getvalue().split('The average SSIM is ')[1].split()[0]
        self.assertAlmostEqual(float(reported), expected_ssim, places=3)

    def test_optimization_with_and_without_vgg(self):
        torch.set_num_threads(1)
        for use_vgg in (False, True):
            for segment_branch in (False, True):
                with self.subTest(vgg=use_vgg, segment=segment_branch):
                    torch.manual_seed(42)
                    network = torch.nn.Conv2d(3, 3, 1)
                    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
                    state = SimpleNamespace(
                        opt={'train': {'ft_tsa_only': 0}},
                        netG=network, optimizer_G=optimizer,
                        var_L=torch.rand(1, 3, 16, 16),
                        real_H=torch.rand(1, 3, 16, 16),
                        l_pix_w=1.0, cri_pix=torch.nn.L1Loss(),
                        ssim_loss=SSIM(), is_vgg_loss=use_vgg,
                        cri_vgg=torch.nn.L1Loss(),
                        seg_map=torch.tensor(1) if segment_branch else None,
                        log_dict={},
                    )
                    before = network.weight.detach().clone()
                    # Exercise the real optimizer method, including backward,
                    # clipping and parameter updates, with a small test network.
                    enhancement_model.optimize_parameters(state, 200)
                    self.assertFalse(torch.equal(before, network.weight))
                    self.assertTrue(all(math.isfinite(v) for v in state.log_dict.values()))
                    self.assertEqual('l_vgg' in state.log_dict, use_vgg)


if __name__ == '__main__':
    unittest.main()
