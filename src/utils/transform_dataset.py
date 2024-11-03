import pandas as pd
from safetensors.torch import save_file
import os
import torch
from PIL import Image
from multiprocessing import Pool
from torchvision.transforms import functional
from safetensors.torch import save_file
from tqdm import tqdm

PATH = "/data/EPIC-KITCHENS-100/images/"
SEGMENT_LENGTH=128
SEGMENT_SEPARATION=64

def create_annotation_file(input_file, output_file, segment_length, segment_separation):
    df = pd.read_csv(input_file)
    df = df[["participant_id", "video_id", "start_frame","stop_frame", "narration", "verb", "noun"]]

    columns = ["participant_id","video_id","start_frame","stop_frame","verb","noun","activity"]
    data = []

    for _, row in df.iterrows():
        frame_count = row["stop_frame"] - row["start_frame"]
        
        if frame_count < segment_length:
            continue
        
        # create fixed sized segments from the original annotations
        for segment_start_frame in range(row["start_frame"], row["stop_frame"] - segment_length + 1, segment_separation):
            
            data.append([
                row["participant_id"],
                row["video_id"],
                segment_start_frame,
                segment_start_frame + segment_length - 1, # startframe + length - 1 frames
                row["verb"],
                row["noun"],
                row["narration"]
            ])
        
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    
def convert_image(fileName:str):
    img = Image.open(fileName).convert('RGB')
    img = img.resize((64, 64), resample=Image.BICUBIC)
    tensor = functional.pil_to_tensor(img)
    return tensor

def process_row(row_tuple):
    _, row = row_tuple
    image_paths = [os.path.join(PATH, 
                                row["participant_id"], 
                                "rgb_frames", 
                                row["video_id"], 
                                f"frame_{str(frame_index).rjust(10, '0')}.jpg") 
                   for frame_index in range(int(row["start_frame"]), int(row["stop_frame"])+1)]
    
    filename = os.path.join(PATH, 
                            row["participant_id"], 
                                "rgb_frames", 
                                row["video_id"], 
                                f"frame_{str(row['start_frame']).rjust(10, '0')}-{str(row['stop_frame']).rjust(10, '0')}.safetensor")

    images = []
    for path in image_paths:
        images.append(convert_image(path))
        
    tensor = torch.stack(images)

    save_file({"tensor": tensor.permute(1, 0, 2, 3).contiguous()}, filename) 
    
if __name__ == "__main__":
    # train
    # create_annotation_file(
    #     "/data/EPIC-KITCHENS-100/annotations/EPIC_100_train.csv",
    #     f"/data/EPIC-KITCHENS-100/annotations/EPIC_100_train_{SEGMENT_LENGTH}_{SEGMENT_SEPARATION}.csv",
    #     segment_length=SEGMENT_LENGTH,
    #     segment_separation=SEGMENT_SEPARATION
    # )
    
    # test
    # create_annotation_file(
    #     "/data/EPIC-KITCHENS-100/annotations/EPIC_100_validation.csv",
    #     f"/data/EPIC-KITCHENS-100/annotations/EPIC_100_test_{SEGMENT_LENGTH}_{SEGMENT_SEPARATION}.csv",
    #     segment_length=SEGMENT_LENGTH,
    #     segment_separation=SEGMENT_SEPARATION
    # )
    
    # preprocess images
    train_file = f"/data/EPIC-KITCHENS-100/annotations/EPIC_100_train_{SEGMENT_LENGTH}_{SEGMENT_SEPARATION}.csv"
    test_file = f"/data/EPIC-KITCHENS-100/annotations/EPIC_100_test_{SEGMENT_LENGTH}_{SEGMENT_SEPARATION}.csv"
    
    for annotation_file in [train_file, test_file]:
        print(annotation_file)
        with open(annotation_file) as f:
            df = pd.read_csv(f, delimiter=',', encoding='utf-8', header=0)
            df = df[["participant_id", "video_id", "start_frame", "stop_frame", "verb", "noun", "activity"]]
            
            with Pool(processes=os.cpu_count()) as pool:  # You can adjust the number of processes as needed
                for _ in tqdm(pool.imap_unordered(process_row, df.iterrows()), total=len(df)):
                    pass