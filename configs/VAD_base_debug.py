_base_ = [
    '../third_party/VAD/projects/configs/VAD/VAD_base_e2e.py'
]

# ============== Debug Overrides ==============
# Reduce resources for local debugging
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # no multiprocessing for debugging
    train=dict(
        # Use mini dataset for quick iteration
        # ann_file=data_root + 'vad_nuscenes_infos_temporal_train.pkl',
    ),
    val=dict(samples_per_gpu=1),
    test=dict(samples_per_gpu=1),
)

# Shorter training for debug
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# More frequent logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# Save checkpoint every epoch
checkpoint_config = dict(interval=1, max_keep_ckpts=2)

# Disable FP16 for easier debugging
# fp16 = None

# Evaluation every epoch
evaluation = dict(interval=1, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')
