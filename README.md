# What is this? 

This repo documents the core components of a SRT generator pipeline using Voxtral to generate the text and WhisperX's forced alignment code to align the subtitles.

## Why is this here?

I am hoping first of all that someone will get use out of this. Secondly I hope that whoever tries to use it, or wants to use it, would open an issue describing what's missing for them, so I know that people are interested in having this solution readily available.

WhisperX is great but I wonder if Voxtral competes or outdoes WhisperX. Problem is, without Voxtral doing forced alignment, you're stuck with a plain .txt file.

## Setup

In truth I do not remember exactly how to set it up. What I know is:

1. You must be on Linux or WSL for Voxtral to run. There is no other way.
2. You need a 12 GB GPU for the model I used. The 4 gb model probably isn't going to outdo WhisperX.
3. You must have a python virtual environment with voxtral installed. If pushed to look further, I can figure out how I installed it on my system.
4. You will need to install  WhisperX to use it, though this can be done from Windows itself.

From there, the files themselves will be the meat and potatoes, the core 90%, of what you need to achieve the following result

## The Results: A Big Deal

Traditionally, language learners strictly use subtitles from WhisperX for offline media. With this pipeline, they now have the opportunity to test the Voxtral model.
