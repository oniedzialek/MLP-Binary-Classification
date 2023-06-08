<h2>Multi-Layer Perceptron Project</h2>
<p>This repository hosts a project for the implementation of a multi-layer perceptron. The project aims to preprocess data, train a neural network on it, and finally evaluate the performance of the trained model.</p>

<h2>Project Structure</h2>
<p>The repository is organized as follows:</p>
<ol>
    <li>RemovingMissingDatas.py: This script is for removing any missing data from the dataset.</li>
    <li>LettersToNumbersConversion.py: This script is responsible for converting categorical values (letters) into numerical values.</li>
    <li>Normalization.py: This script normalizes the converted numerical data.</li>
    <li>Main.py: This is the main script where the multi-layer perceptron model is defined, trained and evaluated.</li>
</ol>

<h2>Setting Up and Running</h2>
<p>Clone the repository: Start by cloning this repository to your local machine using git clone.</p>
<p>Install dependencies: You'll need to install several Python packages to be able to run this project. You can do this by running pip install -r requirements.txt.</p>
<p>Preprocessing: Run the removingMissingDatas.py, lettersToNumbersConversion.py, and Normalization.py scripts to preprocess the data. Run the scripts in the specified order.</p>

<h2>Edit parameters in main.py</h2>
<p>Within the main.py script, you can adjust the following metaparameters:</p>
<ul>
    <li>max_epoch: The maximum number of training epochs.</li>
    <li>err_goal: The desired error goal to achieve.</li>
    <li>disp_freq: The frequency of display updates.</li>
    <li>lr_vec: The learning rate values.</li>
    <li>K1_vec: The first set of node numbers in the hidden layers.</li>
    <li>K2_vec: The second set of node numbers in the hidden layers.</li>
</ul>
<p>Train the Model: Once your data is preprocessed and metaparameters are set, you can run the main.py script to train and evaluate your multi-layer perceptron model.</p>

<h2>The dataset source</h2>
<a href="https://archive.ics.uci.edu/dataset/73/mushroom">https://archive.ics.uci.edu/dataset/73/mushroom</a>
