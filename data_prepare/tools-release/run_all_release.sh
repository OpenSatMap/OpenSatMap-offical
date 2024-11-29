# run all scripts in one script
# Usage: ./run_all.sh
PIC_USE=picuse20trainvaltest
SAVE_FOLDER=picuse20save # save folder, all subfolders will be created in this folder
JSON_FILE=../8.13/all/trainval20.json
TARGET_FOLDER=${SAVE_FOLDER}/pic20gt                  # save the results before cutting train val test
TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/pic20gtsplit # save the results after splitting train val test
ZOOM=20
TRAIN_TXT=picuse20trainvaltest/20train_filenames.txt
VAL_TXT=picuse20trainvaltest/20val_filenames.txt

# PIC_USE=pic1408
# SAVE_FOLDER=project1408merge
# JSON_FILE=satellite_lane_anno_merged.json
# TARGET_FOLDER=${SAVE_FOLDER}/proj19gt
# TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/proj19gtsplit
# ZOOM=19

if [ "$ZOOM" -eq 19 ]; then
    CLIP_SIZE=512
    IMAGE_SIZE=2048
elif [ "$ZOOM" -eq 20 ]; then
    CLIP_SIZE=1024
    IMAGE_SIZE=4096
fi

echo "start running all scripts"
echo ""

# 1. make the files in the desired folders
echo "1. make the files in the desired folders"
rm -rf allpic
rm -rf ${SAVE_FOLDER}
mkdir -p allpic/use
ln -s $PIC_USE/train/* allpic/use
ln -s $PIC_USE/val/* allpic/use
PIC_USE=allpic
echo ""
# # 2. run genertae_GT_multiclass_only-multi.py to get the ground truth
echo "2. run genertae_GT_multiclass_only-multi.py to get the ground truth"
python tools-release/generate_GT_multiclass_only-multi.py --file_name $JSON_FILE \
    --target_folder $TARGET_FOLDER --source_folder $PIC_USE --image_size $IMAGE_SIZE
echo ""
# 3. run python generate_GT_direction_tag.py
echo "3. run python tools-release/generate_GT_direction_tag-multi.py"
python tools-release/generate_GT_direction_tag-multi.py --file_name $JSON_FILE \
    --target_folder $TARGET_FOLDER --source_folder $PIC_USE --image_size $IMAGE_SIZE
echo ""
# 4. generate train val test tag GT
echo "4. generate train val test tag GT"
python tools-release/train_val_test_tag.py --source_folder $TARGET_FOLDER-mask-tag/use --target_folder $TARGET_FOLDER_AFTER_SPLIT \
    --pic_folder ${PIC_USE} --zoom $ZOOM
echo ""
# 5. split other GT train val test
echo "5. split other GT train val test"
python tools-release/train_val_test_gt.py --source_folder $TARGET_FOLDER --target_folder $TARGET_FOLDER_AFTER_SPLIT \
    --pic_folder ${PIC_USE} --zoom $ZOOM
# 6. cut images and GT into small pics
echo "6. cut images and GT"
bash tools-release/run_cut_satellite.sh $TARGET_FOLDER_AFTER_SPLIT ${PIC_USE} $CLIP_SIZE
echo ""
# 7. check the existence of npy files
echo "7. check the existence of npy files"
bash tools-release/check_npy.sh $TARGET_FOLDER_AFTER_SPLIT-cut
echo ""
# 8. find all black images
echo "8. find all black images"
bash tools-release/run_black_images.sh $TARGET_FOLDER_AFTER_SPLIT-cut ${PIC_USE}-cut
echo ""
# 9. remove all black images
echo "9. remove all black images"
bash tools-release/remove_black_image.sh ${PIC_USE}-cut/txt $TARGET_FOLDER_AFTER_SPLIT-cut \
    ${PIC_USE}-cut
echo ""
# 10. move used things to a new path
echo "10. move used things to a new path"
mkdir -p ${SAVE_FOLDER}/final
mv ${PIC_USE}-cut ${SAVE_FOLDER}/final
# mv ${PIC_USE} ${SAVE_FOLDER}
mv $TARGET_FOLDER_AFTER_SPLIT-cut ${SAVE_FOLDER}/final
echo "all scripts have been run"
