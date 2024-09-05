<div align='center'>
  
# LLMs from Scratch

Build a Large Language Model (From Scratch) 
</div>
This repository contains the code and resources for building a large language model (LLM) from scratch, as guided by <b>Sebastian Raschka's book "Build a Large Language Model (from Scratch)."</b> This project aims to demystify the process of creating, training, and fine-tuning LLMs, providing a hands-on approach to understanding the underlying mechanics of these powerful AI models. By following the steps in this repository, you will gain a comprehensive understanding of LLMs and develop your own functional model.


####  All steps to build LLM from scratch in one notebook :
* [llms-from-scratch.ipynb](https://github.com/zeyadusf/LLMs-from-Scratch/blob/main/llms-from-scratch.ipynb)
* Notebook on kaggle [LLMs From Scratch](https://www.kaggle.com/code/zeyadusf/llms-from-scratch)

[![image](https://github.com/user-attachments/assets/97a19a27-8f4a-4d05-8ece-8d91b8136e51)](https://www.manning.com/books/build-a-large-language-model-from-scratch?utm_source=linkedin&utm_medium=organic&utm_campaign=book_raschka_build_12_12_23)



## Chapter 1: Introduction to Large Language Models
*Abstract:*

Chapter 1 of "Build a Large Language Model (from Scratch)" introduces the foundational concepts of large language models (LLMs). It begins by providing a high-level overview of LLMs, their significance in modern AI, and the transformative impact they have had on various applications, such as natural language processing, machine translation, and conversational agents.

The chapter explains the motivation behind creating LLMs from scratch, emphasizing the importance of understanding their inner workings to leverage their full potential. It outlines the scope of the book, detailing the step-by-step approach that will be used to build a functional LLM. 

Large Language Models (LLMs) have revolutionized natural language processing, moving beyond rule-based systems to deep learning-driven methods. LLMs are typically trained in two stages: pretraining on a vast corpus of unlabeled text to predict the next word and fine-tuning on a smaller, labeled dataset for specific tasks. These models are built on the transformer architecture, which uses attention mechanisms to process input sequences. While LLMs like GPT-3 focus on text generation, they also exhibit emergent capabilities like classification and summarization. Fine-tuning LLMs enhances their performance on specialized tasks.

By the end of Chapter 1, readers will have a clear understanding of what LLMs are, why they are important, and what to expect as they progress through the book. This introductory chapter sets a solid foundation for the hands-on journey of building a large language model from the ground up.

> This chapter does not contain any codes.

## Chapter 2: Working with Text Data
*Abstract:*

In this chapter, you'll explore the essential steps involved in preparing input text for training large language models (LLMs). First, you'll learn how to break down text into individual word and subword tokens, which can then be converted into vector representations suitable for input into an LLM. You'll also dive into advanced tokenization techniques such as byte pair encoding (BPE), a method used by prominent models like GPT. The chapter further covers implementing a sampling strategy using a sliding window to generate the necessary input-output pairs for training. Key topics include understanding word embeddings, tokenizing text, converting tokens into token IDs, and the role of special context tokens. Additionally, you'll examine how byte pair encoding improves tokenization and how token embeddings and positional encodings are created and used in LLMs.

By the end of Chapter 2, readers will have a clear understanding of how LLMs handle textual data. Since LLMs cannot process raw text directly, the data must be transformed into numerical vectors, known as embeddings, which are compatible with neural networks. You'll grasp how raw text is tokenized into words or subwords, which are then converted into token IDs. The chapter also explains the significance of special tokens like <|unk|> and <|endoftext|>, which help models understand unknown words or text boundaries. The byte pair encoding tokenizer, utilized in models like GPT-2 and GPT-3, is introduced as an efficient way to handle unknown words. Additionally, you'll see how a sliding window approach can generate training examples and how embedding layers in frameworks like PyTorch map tokens to their vector representations. The importance of positional embeddings, both absolute and relative, in enhancing the model's understanding of token sequences is also emphasized, particularly in the context of OpenAI's GPT models.

> Chapter Codes : 
> [02-Working with Text Data](https://github.com/zeyadusf/LLMs-from-Scratch/tree/main/02-Working%20with%20Text%20Data)
>

## Chapter 3 :Coding Attention Mechanisms:
*Abstract:*

In this chapter, you'll delve into the crucial role of attention mechanisms in neural networks, especially for large language models (LLMs). The chapter begins by exploring why attention mechanisms are used to improve the handling of long sequences in neural models. You'll be introduced to the basic self-attention framework, progressing to more sophisticated versions, and you'll learn how attention helps models capture data dependencies by attending to different parts of the input. The chapter covers implementing a causal attention module that enables LLMs to generate one token at a time, a fundamental feature of autoregressive models. You'll also explore techniques like masking attention weights with dropout to reduce overfitting, and you'll conclude by learning how to stack multiple causal attention modules into a multi-head attention mechanism, a key component of modern transformer architectures.

Key topics include understanding the problem of modeling long sequences, capturing data dependencies using attention mechanisms, and implementing self-attention with trainable weights. You'll learn step-by-step how to compute attention weights and implement a compact self-attention Python class. The chapter also covers the application of causal attention masks to hide future tokens, masking additional attention weights with dropout, and implementing a causal attention class. It concludes with extending single-head attention to multi-head attention, explaining how multiple single-head attention layers can be stacked and how to implement multi-head attention with weight splits.

By the end of the chapter, readers will understand how attention mechanisms transform input elements into enriched context vector representations that incorporate information from all input tokens. You'll learn that a self-attention mechanism calculates the context vector as a weighted sum over the inputs, where attention weights are computed through dot products. While matrix multiplications aren't strictly necessary, they offer a more efficient way to perform these computations. The chapter introduces scaled dot-product attention, where trainable weight matrices help compute intermediate transformations of the inputs—queries, keys, and values. For autoregressive LLMs, a causal attention mask is added to prevent the model from accessing future tokens, and dropout masks are used to reduce overfitting. Finally, multi-head attention, which involves stacking multiple instances of causal attention modules, is discussed, with efficient implementation strategies provided using batched matrix multiplications.

> Chapter Codes :
> [03-Coding attention mechanisms](https://github.com/zeyadusf/LLMs-from-Scratch/tree/main/03-Coding%20attention%20mechanisms)
>

## Chapter 4 : Implementing a GPT model from Scratch To Generate Text 
*Abstract:*

In this chapter, you'll focus on coding a GPT-like large language model (LLM) capable of generating human-like text. The chapter covers several key topics, starting with the implementation of an LLM architecture, including normalizing layer activations to stabilize training. You'll learn about adding shortcut connections in deep neural networks to enhance training effectiveness, and you'll explore the construction of transformer blocks essential for creating GPT models of various sizes. Additionally, you'll compute the number of parameters and storage requirements for GPT models to understand their scale and complexity.

Key topics include coding an LLM architecture, normalizing activations using layer normalization, implementing a feed-forward network with GELU activations, and adding shortcut connections. You'll also cover connecting attention and linear layers in a transformer block and coding the GPT model itself. The chapter concludes with techniques for generating text using the GPT model.

By the end of the chapter, readers will have a clear understanding of how layer normalization helps stabilize training by maintaining consistent mean and variance across layers. You'll learn about shortcut connections, which bypass one or more layers to address the vanishing gradient problem in deep neural networks. Transformer blocks, combining masked multi-head attention modules with feed-forward networks using GELU activations, will be explained as the core structure of GPT models. You'll see how GPT models, with millions to billions of parameters, are built from multiple transformer blocks. The chapter also covers the process of text generation, where the model decodes output tensors into coherent text by predicting one token at a time based on the input context. The importance of training for generating coherent text is highlighted, setting the stage for further exploration in subsequent chapters.

> Chapter Codes :
> [04-Implementing GPT model from scratch](https://github.com/zeyadusf/LLMs-from-Scratch/tree/main/04-Implementing%20GPT%20model%20from%20scratch)
>

## Chapter 5 : Pretraining on Unlabeled Data:
*Abstract:*

In this chapter, you'll learn how to assess and enhance the quality of text generated by large language models (LLMs) during training. Key topics include computing training and validation set losses to evaluate the model's performance, implementing a training function for pretraining the LLM, and saving and loading model weights to continue or resume training. Additionally, you'll explore how to load pretrained weights from OpenAI to leverage existing knowledge and accelerate model development.

You'll start with evaluating generative text models, including how GPT generates text and calculating the associated loss for text generation. The chapter covers calculating training and validation set losses to assess the quality of LLM-generated text. You'll then move on to training the LLM, exploring various decoding strategies to control randomness in text generation, such as temperature scaling and top-k sampling, and modifying the text generation function to suit specific needs. The chapter also addresses practical aspects like saving and loading model weights in PyTorch and incorporating pretrained weights from OpenAI to enhance training efficiency.

By the end of the chapter, readers will understand how LLMs generate text one token at a time, with default behavior being "greedy decoding," where the highest probability token is selected. You'll learn how probabilistic sampling and temperature scaling can influence the diversity and coherence of generated text. Training and validation set losses will be used to evaluate text quality, and you'll grasp the process of pretraining an LLM, including using a standard training loop with cross-entropy loss and the AdamW optimizer. The chapter emphasizes that pretraining an LLM on a large dataset can be resource-intensive, making the option to load pretrained weights from OpenAI a valuable alternative to expedite model development.

> Chapter Codes :
> [05-Pretraining on Unlabeled Data](https://github.com/zeyadusf/LLMs-from-Scratch/tree/main/05-Pretraining%20on%20Unlabeled%20Data)
>
