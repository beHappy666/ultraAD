_base_ = [
    '../third_party/VAD/projects/configs/VAD/VAD_tiny_e2e.py'
]

# ============== Tiny Debug Config ==============
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val=dict(samples_per_gpu=1),
    test=dict(samples_per_gpu=1),
)

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=1, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')
