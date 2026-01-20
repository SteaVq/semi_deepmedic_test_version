基于 FixMatch 将deepmedic项目重构为半监督版本。修改了采样逻辑,增加有标签样本数量这个参数（train_config里面），引入高斯噪声作为强增强手段，并新增了一致性正则化 Loss 以提升训练效果。
