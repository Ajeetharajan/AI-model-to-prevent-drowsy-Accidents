import os

# Path to mouth dataset
no_yawn_path = 'extracted_mouth_frames/no_yawn'
yawn_path= 'extracted_mouth_frames/yawn'

# Count images
yawn_count = len([f for f in os.listdir(yawn_path) if os.path.isfile(os.path.join(yawn_path, f))])
no_yawn_count = len([f for f in os.listdir(no_yawn_path) if os.path.isfile(os.path.join(no_yawn_path, f))])

total = yawn_count + no_yawn_count

print(f"Yawn images: {yawn_count}")
print(f"No Yawn images: {no_yawn_count}")
print(f"Total mouth images: {total}")
