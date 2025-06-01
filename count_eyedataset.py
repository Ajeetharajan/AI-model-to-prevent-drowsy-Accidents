import os

# Path to eye dataset
open_eye_path = 'dataset/open_look'
closed_eye_path = 'dataset/closed_look'

# Count images
open_count = len([f for f in os.listdir(open_eye_path) if os.path.isfile(os.path.join(open_eye_path, f))])
closed_count = len([f for f in os.listdir(closed_eye_path) if os.path.isfile(os.path.join(closed_eye_path, f))])

total = open_count + closed_count

print(f"Open eye images: {open_count}")
print(f"Closed eye images: {closed_count}")
print(f"Total eye images: {total}")
