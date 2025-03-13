import time
import warnings

import torch

import mmcv
from mmcv.runner import IterBasedRunner, IterLoader
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info


@RUNNERS.register_module()
class IterOVDRunner(IterBasedRunner):

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1

    def run(self,
            data_loaders,
            workflow,
            max_iters=None,
            **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        # assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')
        for i in range(len(data_loaders)):
            print(len(data_loaders[i]))

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[self.iter % len(data_loaders)], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

# 为了使本文件生效，需要修改5处代码：
# 1. tools/train.py，在里面需要加入一个build_dataset
# 2. mmdet/apis/train.py, 在里面修改了build_dataloader的方式, 并且修改了第231行by_epoch的eval参数
# 3. 增加本文件，本文件在mmcv的IterBasedRunner的基础上，对多个dataloader进行循环调用，见第89行
# 4. 修改mmdet/datasets/builder.py文件，使得本runner也能够使用infinitesampler
# 5. 对应修改config文件，修改runner为IterOVDRunner，在data下面，除了train之外，可以增加一个train_uncurated，两个数据集可以使用不同的batchsize，用列表隔开
# 6. 修改了tools/test.py中274行丢掉by_epoch的参数


