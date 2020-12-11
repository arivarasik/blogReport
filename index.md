---
layout: default
title: Blog Post
tagline: Speech Separation using Audio-Visual Features
description: Blog post for EE 380L Data Mining Project
---

***Alexander Fiore, Alexander Phillips, Arivarasi Kesawaram, Dae Yeol Lee, Dan Jacobellis***

[comment]: # (Abstract: 1-2 Paragraphs)

The visual component of human speech provides rich information about the acoustic signal that can be exploited for source separation. For our project, we train a model to estimate the time-frequency audio mask that separates two speakers in a single channel recording that utilizes the visual information of the speech. Over 5000 videos were retrieved from the [AVspeech][2] dataset and combined into pairs to form speech mixtures and ground truth labels. To our knowledge, this project is the first publicly available end-to-end implementation for this task.
![Cocktail Party Effect](/img/cocktail.png)
This scenario is what we call a ‘cock-tail party phenomenon’. It describes the ability to focus one’s listening attention on a single speaker among a mixture of conversations and background noises, ignoring other conversations. Our brain does this Audio-source separation all the time without us even realizing it. There is a lot that is unknown about the processes in the human auditory system that achieve this, and developing computational methods to replicate it remains an open problem. The binaural processing of the human auditory system in combination with visual cues allows us to effectively focus our attention on a single speaker when multiple speakers and other noise are present. When audio containing multiple speakers is recorded digitally and combined into a single channel (e.g. the recent presidential debates), the visual and spatial component of the acoustic field is lost, and it becomes much harder for a listener to understand the speakers individually. So, this is a difficult problem to solve in the world of signal processing.
In this project, we have built a Deep Neural Network (DNN) model, which is a joint audio-video separation model that can take in both auditory and visual features extracted from a video and decomposes the input mixed audio track into distinct output audio tracks, one for each speaker in the video. We train the model to estimate a time-frequency audio mask that separates two speakers in a single channel recording. This mask when applied to the input mixed audio, will filter out the desired individual track. 
This is what a high-level view of what the model looks like.
![Looking to listen](/img/looking-to-listen.PNG)

[comment]: # (Introduction & Background / Problem being addressed and why it’s important / Related work / approach and rationale / contribution or novel characteristics)
# 1. Introduction and Motivation
Imagine you are at a holiday party, or a gathering with friends and you are engaged in a discussion with some of your friends. A lot of people are talking simultaneously, but you are not paying attention to those conversations. But then, you hear someone say your name suddenly and you realize a group of people having a discussion behind you. Now you are unable to concentrate on the discussion you were having previously, because you are curious to know what the other people are talking about you. You weren’t deliberately eavesdropping on the group’s conversation, you just happened to hear your name. Is it even possible to unconsciously eavesdrop? 


# Audio separation
The binaural processing of the human auditory system in combination with visual cues allows us to effectively focus our attention on a single speaker when multiple speakers and other noise are present. When audio containing multiple speakers is recorded digitally and combined into a single channel (e.g. the recent presidential debates), the visual and spatial component of the acoustic field is lost, and it becomes much harder for a listener to understand the speakers individually. 

The exact mechanisms of the human auditory system to separate speech (the cocktail party effect) are not fully understood and developing computational methods to replicate it remains an open problem. State of the art methods that use audio only, also known as blind separation methods, still leave much to be desired, which motivates the use of secondary information, either visual or spatial. Since recorded audio is increasingly accompanied by video (but not by multi-channel audio), exploting visual information is more convenient and useful. Use of the visual signal is further motivated further by its known power, as evidenced by human lip reading.

# Related Work
To our knowledge, the current state of the art for audio-visual separation is a [deep network model developed by google research][1] which was later accompanied by the [AVSpeech][2] dataset. While its results are impressive, the implementation is proprietary and many of the details of the processing pipeline are left ambiguous. Regardless, it provides an excellent learning model architecture and introduces several novel techniques which we have adopted. To our knowledge, our implementation is the first publicly available, end-to-end audio-visual speech separation model.

# Overview of architecture
The model explored in our project is a deep neural network (DNN) trained using face embeddings
generated by the [FaceNet][3] model and the mixture of audio streams from two videos. The output of the model is a time-frequency audio mask to be used for audio source separation. The adopted audio-visual network architecture consists of both convolutional and recurrent layers. 

The visual pre-processing consists of (1) a face detection model, (2) rejection of videos with occluded faces (3) resampling of cropped faces to a uniform size, and (4) per-frame application of the FaceNet model to create a sequence of face embeddings. The audio pre-processing consists of (1) mixing the audio from a pairs of videos containing speech, (2) applying a time-frequency transform, and (3) constructing the ideal mask to use as the ground truth in training.

YouTube videos referenced by URLs in the AVspeech dataset were downloaded and preprocessed in pairs to generate training data corresponding to a two-speaker scenario. The final model was trained on approximately 2500 mixtures from 5000 YouTube videos containing unique speakers

A variant of the structural similarity measure used in image quality assessment was used as the loss between the predicted and ideal audio masks during the training. The network was trained using a consumer GPU on a personal workstation iterating over the full dataset in several epochs. It produces speech separation masks which show some similarity to the ideal masks, but the error remains quite high judging by listening to the reconstructed audio streams.

* \[Data Collection/Description\]
  * \[Relevant Characteristics\]
  * \[Source(s)\]
  * \[Methods of acquisition\]
  
# Audio Features
The STFT (Short-time fourier transform), was chosen to represent the audio as a joint time-frequency distribution. The STFT is one of the simplest ways to construct a time-frequency distribution and is ubiquitous for audio signal processing. It is constructed by taking the discrete fourier transform (DFT) over a sliding time window. For example, you might take the DFT over a 10 ms window, giving you an estimate of the signal spectrum locally for that 10 ms window, then repeat this for successive windows to construct a matrix.

![STFT](/img/stft.png)

Usually the magnitude of the DFT values are used, in which case it is called the magnitude spectrogram, but in our case we have left the signal as separate real and imaginary parts which allows the inverse operation to be performed to exactly recover the original signal.

![STFT](/img/real_imag.png)

<audio controls src="https://danjacobellis.github.io/AV-speech-separation/audio/ideal_stream0.wav" type="audio/wav">

The motivation for applying a time frequency transform is its effect on the sparsity of the signal, which in turn affects separability. For speech in particular, where the primary signal is produced by vibration of the vocal folds, a large fraction of the energy is concentrated at the fundamental frequency of the vibration and its harmonics. These frequencies change over time, which is why a joint representation is necessary, but result is that the important information characterizing the speech is contained in a small area of the distribution, making it easy to isolate.

Historically, speech separation is usually performed by defining the ideal mask in terms of just the magnitude spectrogram. Recent work has demonstrated better results can be achieved using a complex mask.

Range compression of the audio helps stabilize the training of the network by preventing outlying values from dominating the gradient. Usually a spectrogram is viewed in decibels which compresses its range, but since we’ve left the values as complex and can be negative, we instead compress the range using a hyperbolic tangent.

  
  
* \[Data Pre-Processing & Exploration\]
  * \[Feature engineering/selection\]
  * \[Relevant Plots\]
* \[Learning/Modeling\]
  * \[Chosen models and why\]
  * \[Training methods (validation, parameter selection)\]
  * \[Other design choices\]
* \[Results\]
  * \[Key findings and evaluation\]
  * \[Comparisons from different approaches\]
  * \[Plots and figures\]
* \[Conclusion\]
  * \[Summarize everything above\]
  * \[Lessons learned\]
  * \[Future work - continuations or improvements\]
* \[References\]
* \[Relevant project links (i.e. Github, Bitbucket, etc…)\]


[1]:https://looking-to-listen.github.io
[2]:https://looking-to-listen.github.io/avspeech/
[3]:https://github.com/davidsandberg/facenet
