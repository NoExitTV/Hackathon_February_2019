# Hackathon_February_2019
Machine Learning hackathon held at LTU. This was initialized for the group that trains networks on rectangular images.

# Idea
Use resnet/vgg/alexnet that have been trained on IMAGENET using rectangular images. The idea is to observe the original aspect ration as much as possible for the documents (rectangular) to loose as little information as possible.

# How to execute?
Run with commant:  
'''
python rectangular-benchmark.py 2>&1 | tee -a outputs/$(date +%F_%T).log
'''