Ref: 
https://www.youtube.com/watch?v=L3BTG8ETY_Y
https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training

CLIP: Contrastive Language-Image Pre-training (CLIP) is a technique for training a pair of neural network models, one for image understanding and one for text understanding, using a contrastive objective. It is an image classification model technique from OpenAI. 
1. The image encoder could be ResNet, ConvNeXt, or visiion transforemr.
2. The text encoding models used in CLIP are typically Transformers.

As in CLIP, the dataset classifier from label text is done only once at before inference, it's inference latency is low.
Also, since only one new image has to be passed through the image encoder to get image vector representation at inference, it is used for **zero shot prediction** as well.


CLIP (Contrastive Language-Image Pre-training) is considered a **multimodal foundation model or vision-language model (VLM)**. It acts as a bridge between visual and textual data, allowing AI to understand images in the context of natural language. 

Here is a breakdown of why CLIP is considered multimodal, and where it differs from a traditional "Multimodal LLM":

**Why CLIP is Multimodal**
Dual-Encoder Architecture: CLIP consists of two separate models: one to process text (Text Encoder) and one to process images (Vision Encoder/ViT).
Shared Representation Space: CLIP is trained on 400 million image-text pairs to embed images and text into the same mathematical vector space.
Cross-Modal Understanding: Because they share a space, CLIP can determine how well a text description matches an image, enabling tasks like zero-shot image classification and image-text retrieval. 

**CLIP vs. Multimodal LLMs (e.g., LLaVA, GPT-4V)**
While CLIP is multimodal, it is typically classified as an encoder rather than a "Large Language Model" (LLM) itself. 
CLIP: Produces embeddings (numerical representations) that tell you how similar an image is to a text. It does not generate text or reason extensively on its own.
Multimodal LLM: Uses a model like CLIP as a "vision backbone" to extract features, which are then passed into a generative language model (like Llama or GPT) to describe the image, answer questions, or reason. 

**Key Takeaways**
Role: CLIP is the foundational vision encoder that enables LLMs to "see".
Modern Enhancements: Techniques like LLM2CLIP are currently being developed to enhance CLIP by replacing its text encoder with a full LLM to improve its ability to understand long, complex captions. 

In summary, CLIP is a **multimodal vision-language model** used for aligning images and text, often acting as the visual backbone inside broader Multimodal LLM systems.
