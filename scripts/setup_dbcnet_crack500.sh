#!/bin/bash
# Setup DBCNet data for CRACK500_resize on server
DS="/root/autodl-tmp/ComparisonModel/dataset_8_2/CRACK500_resize"
DBCNET="/root/autodl-tmp/ComparisonModel/DBCNet"

mkdir -p "$DBCNET/data/CRACK500_resize"
cd "$DBCNET/data/CRACK500_resize"

# Symlinks
ln -sfn "$DS/train/images" train_img
ln -sfn "$DS/train/annotations" train_lab
ln -sfn "$DS/test/images" test_img
ln -sfn "$DS/test/annotations" test_lab

# Generate train.txt
> train.txt
for f in $(ls train_img/*.png 2>/dev/null | sort); do
    bn=$(basename "$f")
    echo "train_img/$bn train_lab/$bn" >> train.txt
done

# Generate test.txt
> test.txt
for f in $(ls test_img/*.png 2>/dev/null | sort); do
    bn=$(basename "$f")
    echo "test_img/$bn test_lab/$bn" >> test.txt
done

echo "train.txt: $(wc -l < train.txt) lines"
echo "test.txt: $(wc -l < test.txt) lines"
echo "Done"
