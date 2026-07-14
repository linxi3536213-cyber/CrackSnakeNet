import torch

class Config:

    # ========== Data Config ==========
    # 数据集根目录 (7:1.5:1.5 划分, 去重519张, 子目录: images/, annotations/, centerlines/, 各含 train/val/test/)
    DATASET_ROOT = r"/home/andy/desktop/CrackDectection/CrackMambaTransformerNet/ablation/CrackMambaTransformerNetTest3.3TriFFMplus/ComparisonModel/dataset/DeepCrack"
    IMAGE_DIR = "images"
    MASK_DIR = "annotations"
    CENTERLINE_DIR = "centerlines"
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "val"
    INVERT_MASK = True          # True = 黑色是裂缝, 白色是背景 (omnicrack30k格式)
    IMAGE_SIZE = (512, 512)

    # ========== DataLoader Config ==========
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2
    
    # ========== Data Augmentation Config ==========
    AUGMENTATION_STRENGTH = 'medium'  # 中等增强提升泛化

    # ========== Model Config ==========
    INPUT_CHANNELS = 3
    NUM_CLASSES = 1
    ENCODER_FIRST_DOWNSAMPLE_STRIDE = 1

    @staticmethod
    def expand_stage_blocks(blocks, stage_count: int):
        if len(blocks) == 1:
            return blocks * stage_count
        if len(blocks) < stage_count:
            raise ValueError(f"Stage block length {len(blocks)} does not match stage count {stage_count}")
        return blocks[:stage_count]

    # 边界分支（消融实验证明关闭更好）
    USE_EDGE_HEAD = False
    EDGE_LOSS_WEIGHT = 0.0
    EDGE_AUX_LOSS_WEIGHT = 0.0
    EDGE_THRESHOLD = 0.1

    # 中心线召回损失：迫使模型覆盖细裂缝中心线（轻量，不干扰主损失）
    CENTERLINE_LOSS_WEIGHT = 0.15

    # 深监督
    USE_AUX_OUTPUTS = True
    AUX_STAGE_INDICES = (0, 1, 2)  # 4 stage 对应 3 个 decoder，选 3 层深监督

    # Transformer配置
    TRANSFORMER_WINDOW_SIZE = 7
    TRANSFORMER_GLOBAL_POOL = 4

    USE_GRADIENT_CHECKPOINTING = False

    # ========== Model Architecture Config ==========
    # 4-stage + 降低复杂度 + 双向交叉注意力 + 多尺度 Mamba + 蛇形卷积
    MODEL_CONFIGS = {
        "depths": (1, 1, 2, 2),
        'widths': (48, 96, 192, 384),
        'k': 7,  # 方向卷积核大小
        'd_state': 16,
        'drop_path_rate': 0.1,
        'dropout': 0.1,
        'attn_drop': 0.05,
        'proj_drop': 0.1,
        'scan_directions': 8,  # 扫描方向数 (2/4/6/8 用于消融实验)
    }

    # ========== Training Config ==========
    EPOCHS = 600
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3

    ACCUMULATE_GRAD_BATCHES = 8
    GRADIENT_CLIP_VAL = 0.5
    METRIC_THRESHOLD = 0.5

    LR_WARMUP_EPOCHS = 30
    LR_MIN = 5e-8

    # ========== Batch-level Augmentation ==========
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.0

    # ========== Loss Config ==========
    LOSS_CONFIG = {
        'loss_types': ['focal', 'dice'],
        'loss_weights': {
            'focal': 0.3,
            'dice': 0.7,
        },
        'focal_alpha': 0.65,
        'focal_gamma': 2.5,
    }

    # ========== Early Stopping Config ==========
    EARLY_STOPPING_CONFIG = {
        'monitor': 'val_iou',
        'patience': 80,
        'mode': 'max',
        'verbose': True,
        'min_delta': 0.001
    }

    # ========== Validation Config ==========
    VAL_CHECK_INTERVAL = 1.0
    LOG_EVERY_N_STEPS = 20
    
    # ========== Inference Config ==========
    INFERENCE_INPUT_FOLDER = '/home/andy/desktop/CrackDectection/dataset/ConcreteCrack/sample/images'
    INFERENCE_OUTPUT_FOLDER = 'results'
    INFERENCE_NUM_SAMPLES = 5
    INFERENCE_THRESHOLD = 0.5
    INFERENCE_MASK_FOLDER = '/home/andy/desktop/CrackDectection/dataset/ConcreteCrack/sample/masks'

    THRESHOLD_SEARCH = {
        'thr_min': 0.10,
        'thr_max': 0.90,
        'thr_step': 0.01,
        'output_dir': 'results',
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
