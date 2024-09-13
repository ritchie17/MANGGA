import os
import subprocess

def combine_videos(input_folder, output_file):
    # Ensure the input folder path ends with a slash
    if not input_folder.endswith('/'):
        input_folder += '/'

    # Get all video files in the folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    video_files.sort()  # Sort to maintain a specific order

    # Create a text file listing all videos to be combined
    with open('file_list.txt', 'w') as file_list:
        for video_file in video_files:
            file_list.write(f"file '{os.path.join(input_folder, video_file)}'\n")

    # Use ffmpeg to combine the videos
    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'file_list.txt', '-c', 'copy', output_file]
    subprocess.run(command)

    # Clean up the temporary file list
    os.remove('file_list.txt')

# Example usage
input_folder = '/mnt/f/For Copywrite (videos)/Videos Combined'
output_file = '/mnt/f/For Copywrite (videos)/Videos Combined/VideoCombined.mp4'
combine_videos(input_folder, output_file)
