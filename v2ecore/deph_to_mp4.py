import scipy.io
import numpy as np
import cv2
import glob
import os 
def mat_to_mp4(file: str) -> None:
    video_path = file
    mat_file_path = video_path.split(".")[0] +"_depth.mat"
    output_depth_video_path = video_path.split(".")[0] +"_depth.mp4"
    # test if output file already exists
    if "depth" in file or os.path.exists(output_depth_video_path):
        return None
    
    video_path = file
    
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)

    # Load the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define a threshold to remove extreme values (e.g., 10e10)
    max_valid_depth = 10000  # Adjust this threshold as needed

    all_depth_values = []
    for key in mat_data:
        if key.startswith("depth_"):
            frame = -mat_data[key].astype(np.float32)
            frame[np.abs(frame) > max_valid_depth] = np.nan
            valid_values = frame[~np.isnan(frame)]
            all_depth_values.extend(valid_values.flatten())

    if all_depth_values:
        global_min = np.min(all_depth_values)
        global_max = np.max(all_depth_values)
    else:
        global_min, global_max = 0, 1  # Default to avoid division by zero

    # Create a video writer for the depth-only video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    depth_out = cv2.VideoWriter(output_depth_video_path, fourcc, fps, (frame_width, frame_height))

    # Prepare window for side-by-side comparison
    while cap.isOpened():
        ret, video_frame = cap.read()
        if not ret:
            break
        
        # Get the current frame index
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        depth_key = f'depth_{frame_idx}'
        
        if depth_key in mat_data:
            depth_frame = -mat_data[depth_key].astype(np.float32)
            
            # Ignore extreme values by setting them to NaN
            depth_frame[np.abs(depth_frame) > max_valid_depth] = np.nan
            
            # Normalize using global min/max
            if global_max > global_min:
                depth_frame = 255 * (depth_frame - global_min) / (global_max - global_min)
            
            # Replace NaN with zero (background should be black)
            background_mask = np.isnan(depth_frame) | (depth_frame == 0)
            depth_frame = np.nan_to_num(depth_frame, nan=0).astype(np.uint8)
            
            # Apply colormap for better depth visualization
            depth_colored = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            
            # Ensure background remains dark (black or gray)
            depth_colored[background_mask] = [50, 50, 50]  # Dark gray background
            
            # Resize depth frame to match the video frame size
            depth_resized = cv2.resize(depth_colored, (frame_width, frame_height))
            
            # Write the depth frame to the depth video
            depth_out.write(depth_resized)
            
            # Concatenate the original and depth frames side by side
            combined_frame = np.hstack((video_frame, depth_resized))
            
            # Show the frames
            cv2.imshow('Video and Depth Side-by-Side', combined_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
            exit()
    # Release video writers and windows
    cap.release()
    depth_out.release()
    cv2.destroyAllWindows()

    print(f"Depth video saved as: {output_depth_video_path}")

if __name__ == "__main__":
    data_path = "/home/martin-barry/Downloads/cmu/"
    files = glob.glob(os.path.join(data_path, "**/*.mp4"), recursive = True)
    for file in files:
        mat_to_mp4(file)