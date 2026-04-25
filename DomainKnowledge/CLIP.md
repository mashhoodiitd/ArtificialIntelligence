Ref: 
https://www.youtube.com/watch?v=L3BTG8ETY_Y
https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training

CLIP: Contrastive Language-Image Pre-training (CLIP) is a technique for training a pair of neural network models, one for image understanding and one for text understanding, using a contrastive objective. It is an image classification model technique from OpenAI. 
1. The image encoder could be ResNet, ConvNeXt, or visiion transforemr.
2. The text encoding models used in CLIP are typically Transformers.

As in CLIP, the dataset classifier from label text is done only once at before inference, it's inference latency is low.
Also, since only one new image has to be passed through the image encoder to get image vector representation at inference, it is used for **zero shot prediction** as well.
