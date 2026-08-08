# Reproduction Guide: Accentor + StreamSpeech (NeuralSpeechLab)

This document provides a comprehensive guide to reproducing a high-quality, linguistically accurate, and computationally efficient text-to-speech (TTS) system. It unites the concepts of the **Accentor** network, which solves complex lexical stress ambiguities using explicit contextual modeling, and **StreamSpeech**, an optimized architecture designed for faster-than-real-time, low-latency synthesis on resource-constrained devices. 

By separating linguistic stress prediction from acoustic feature generation, this pipeline enables highly accurate speech synthesis without requiring an impractically large end-to-end TTS acoustic model.

## Phase 1: Data Compilation and Annotation
To train a model capable of predicting lexical stress based on context, you must first compile and automatically annotate a massive, lexically diverse text corpus.

1.  **Gather Raw Data:** Collect extensive audio-text pairs. For the Bulgarian model, researchers used 11.7K hours of speech from Bulgarian Parliament plenary session videos (2010–2018) and professional audiobook recordings paired with texts from the Chitanka library.
2.  **Automatic Alignment:** Use a hybrid DNN-HMM Automatic Speech Recognition (ASR) system to align the audio with the text. Use a phoneme-level Levenshtein distance to handle numbers and abbreviations, and extract only the sentences where the audio perfectly matches the transcription. 
3.  **Automatic Stress Annotation:** 
    *   Train a time delay deep neural network ASR acoustic model capable of differentiating between stressed and unstressed vowels using a phonetic system with separate phonemes for both. 
    *   For each aligned utterance, construct a decoding graph that encodes all possible canonical stress patterns for the words in that sentence.
    *   Decode the audio to classify and annotate the vowels based on the speaker's actual pronunciation.
4.  **Dictionary Extension:** To maximize lexical coverage, add words with unambiguous stress patterns from a standard dictionary as separate utterances. 

## Phase 2: Building and Training the Accentor Model
The Accentor is a neural network designed to ingest raw text and output text explicitly annotated with lexical stress marks. 

1.  **Model Architecture:** Construct a network consisting of an embedding layer, 6 Feed-Forward Transformer (FFT) blocks, and a final predictor. The FFT blocks utilize multi-head self-attention to process the entire input sequence simultaneously, which is required to resolve contextual homographic ambiguities.
2.  **Dimensionality:** Set the embedding dimension and hidden size of the self-attention to 256, utilizing 4 attention heads of dimension 128.
3.  **Efficiency Optimizations:** To reduce the model size and increase inference speed, replace standard 1D convolutions in the network with depth-wise separable convolutions. 
4.  **Vocabulary Definition:** Define the input vocabulary based on the target language. For Bulgarian, use 41 symbols: 30 letters, 10 punctuation marks, and a word boundary symbol. 
5.  **Training:** Train the model using binary cross-entropy loss between the predicted logits and the target binary characteristic vector of stressed vowels. Train on 90% of your compiled corpus using a batch size of 512.

## Phase 3: Building and Training the StreamSpeech Acoustic Model
The StreamSpeech model acts as a "controllable acoustic model" that accepts the stress-annotated text from the Accentor and rapidly generates Mel spectrograms.

1.  **Modify the Input Vocabulary:** Expand the acoustic model's input vocabulary to include the base symbols used by the Accentor, plus additional symbols representing the newly predicted stressed vowels (e.g., 47 symbols total for the Bulgarian model).
2.  **Optimize the FastSpeech 2 Encoder:** Use the standard FastSpeech 2 encoder architecture to model long-term prosody, but replace the vanilla convolutions in the FFT blocks and predictors with depth-wise separable convolutions to reduce the parameter count by a factor of 3.8.
3.  **Construct the Convolutional Decoder:** Remove the standard transformer-based FastSpeech 2 decoder, which causes high latency due to global attention. Replace it with a lightweight, streamable convolutional decoder consisting of a stack of 5 Conformer blocks (a convolutional module followed by a feed-forward module). Set the hidden dimension to 256 and the convolution kernel size to 31.
4.  **Training:** Train the acoustic model on a standard-sized, high-quality TTS dataset (e.g., 23 hours of speech) where the transcriptions have been explicitly annotated with lexical stress marks using the pipeline from Phase 1.

## Phase 4: Building the Vocoder (LPCNet)
The final step requires an optimized vocoder to convert the 80-dimensional Mel spectrograms from StreamSpeech into 24 kHz audio waveforms. 

1.  **Increase Sampling Rate:** Modify the standard LPCNet architecture to synthesize 24 kHz speech by extracting 24 linear prediction coefficients from the input Mel spectrograms (which represent 10.7-ms frames). Increase the receptive field using a stack of five 1x3 1D convolutions with 256 channels.
2.  **Multi-Band Processing:** Modify the vocoder to predict audio signals in parallel rather than sample-by-sample. Implement multi-band processing using 4 frequency bands and analysis/synthesis FIR filters of order 64.
3.  **Hierarchical Softmax:** To reduce the computational load of the dual fully-connected layers, represent the output distribution of each frequency band as an 8-level binary tree. This reduces the evaluated outputs during sampling from 256 to 8.

## Phase 5: Pipeline Inference execution
To run the fully reproduced model for text-to-speech synthesis:

1.  Pass raw input text through the **Accentor model**, which will output the exact same text annotated with explicit lexical stress marks based on whole-sentence context.
2.  Pass this annotated text into the **StreamSpeech acoustic model**. To achieve the optimal balance between low latency and real-time factor (RTF) efficiency, run the decoder in streaming mode parameterized to process inputs simultaneously at a `Rate = 6`.
3.  Pass the resulting generated Mel spectrograms into the optimized **multi-band LPCNet vocoder** to produce the final highly intelligible, low-latency speech waveform.