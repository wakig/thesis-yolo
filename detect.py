import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    plot_all_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    anchors = config.ANCHORS
    device = config.DEVICE

    plot_all_examples(model, test_loader, 0.6, 0.5, scaled_anchors)

if __name__ == "__main__":
    main()


# x = 'COCO/images/000000000036.jpg'
# x = x.to(config.DEVICE)
# with torch.no_grad():
#     predictions = model(x)

# print(predictions)

# for idx, (x, y) in enumerate(test_loader):
    # x = x.to(config.DEVICE)
    # with torch.no_grad():
    #     out = model(x)
    #     print(out)