"""Tests for long-horizon action history and episode-level paired statistics."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parents[1]/'examples/wanvideo/model_training'
sys.path.insert(0,str(DIRECTORY))
import medium_orca_action_probe as medium
import probe_orca_action_conditioning as base


class LongHorizonTests(unittest.TestCase):
    def test_future_intervention_history_reaches_later_chunks(self):
        real=torch.arange(29*58,dtype=torch.float32).reshape(29,58)
        hold=torch.full((58,),-37.)
        command=base.intervention(real,hold,-real,'hold')
        first=medium.chunk_actions(command,0)
        torch.testing.assert_close(first[:5],real[:5])
        torch.testing.assert_close(first[5:],hold.expand(8,58))
        for chunk in (1,2):
            later=medium.chunk_actions(command,chunk)
            torch.testing.assert_close(later[0],real[0])
            # No ground-truth action history may reappear after intervention.
            torch.testing.assert_close(later[1:],hold.expand(12,58))

    def test_reverse_applies_to_complete_future_without_reversing_observations(self):
        real=torch.arange(29,dtype=torch.float32)[:,None].expand(29,58)
        command=base.intervention(real,torch.zeros(58),real,'reverse')
        expected_futures=[list(range(28,20,-1)),list(range(20,12,-1)),list(range(12,4,-1))]
        for chunk,expected in enumerate(expected_futures):
            np.testing.assert_equal(medium.chunk_actions(command,chunk)[5:,0].numpy(),expected)
        torch.testing.assert_close(command[:5],real[:5])

    def test_rollout_metric_scores_each_chunk_and_shared_motion_region(self):
        last=np.zeros((2,2,3),dtype=np.uint8)
        truth=np.zeros((24,2,2,3),dtype=np.uint8);truth[:,0,0]=255
        generated=truth.copy();generated[8:16,0,0]=0
        r=medium.metric_rows(dict(id='a',episode=1,split='val_data'),9,'real',generated,truth,last)
        by_chunk={v['chunk']:v for v in r}
        self.assertAlmostEqual(by_chunk['all']['motion_region_mse'],1/3)
        self.assertEqual(by_chunk['2']['motion_region_mse'],1)
        self.assertEqual(by_chunk['1']['motion_region_mse'],0)
        self.assertEqual(by_chunk['3']['motion_region_mse'],0)

    def test_statistics_weight_episodes_not_correlated_frames(self):
        records=[]
        for ep,count,delta in [(1,100,1.),(2,1,-3.)]:
            for i in range(count):
                for variant,err in [('real',10.),('hold',10.+delta)]:
                    records.append(dict(id=f'{ep}-{i}',episode=ep,seed=1,variant=variant,mse=err))
        result=medium.paired(records,'mse','hold')
        self.assertEqual(result['delta'],-1.)
        self.assertEqual(result['episode_win_rate'],.5)
        self.assertLess(result['ci95'][0],0)


if __name__=='__main__':
    unittest.main()
