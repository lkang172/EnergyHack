# WattsTheMatter

# Demo
'https://youtu.be/VYcbI79A4LE?si=EKf8YcGQorSxBpM9'

# Inspiration
As the use of AI becomes exponentially widespread in all industries, the massive environmental effects of training machine learning models are often left in the dark. Hearing about how ChatGPT is set to consume more energy than Sweden, Argentina, or the Netherlands by 2027, we were driven to create an easy-to-use tool to predict a model's energy consumption without having to run it first. This would allow the user to optimize their code to achieve the lowest possible environmental impact, all before using energy to run the model in the first place.

# What it does
WattsTheMatter is a web app that takes input from the user in the form of raw code for any type of machine learning model, the batch and dataset size, the GPU used, and the region of the world that the user is training in. It will then calculate a close approximation of how much energy is consumed and how much carbon is emitted, so that the user can modify their model before they ever run it and consume resources.

# How we built it
First, we researched different types of machine learning models, and in particular the different types of layers involved in neural networks. We then investigated how each type of layer worked, like how they did linear combinations between nodes or how many operations were done between layers.

Since the source of energy expenditure when training ML models comes overwhelmingly from these multiply and accumulate operations (MACs), we created an algorithm to extract known layer types from the input code (using the AST library) and then calculate their specific MAC value for one pass of the model. Then, we calculated the FLOPs (floating point operations), which is a unit of measure to calculate the energy usage of the model.

Once we calculated the total FLOPs for one iteration of the model, we then divided that by a unique FLOPs per watt value dependent on the user's GPU, which determines how many FLOPs can be executed for each watt. Then we apply the batch size and dataset size to find the final amount of watts used. We also put this value into context so that the user can fully comprehend the scale of their ML training.

