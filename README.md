#Retail Queue Analysis
Welcome! 
This is a machine learning project that analyses video footages of retail shop queues and calculates their operational efficiency based on several metrics.
The metrics include: Peak Time, Average Wait, Congestion Rate, General Efficiency.
The computer vision model uses pre-trained weights of the state-of-the-art yolov8 to detect people in queues and calculate their wait time and writes the results to a csv file.
This csv file is then used by the secondary script to calculate the metrics.
The results are visualized on a Flask App.
