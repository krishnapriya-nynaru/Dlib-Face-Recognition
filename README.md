# Dlib Face Recognition

## Introduction

Dlib Facial Recognition is a state-of-the-art facial recognition system that leverages the capabilities of the Dlib library. It combines advanced machine learning techniques and efficient algorithms to detect, recognize, and process faces in real-time. This system is widely used in various applications, including security, surveillance, user authentication, and interactive systems, due to its accuracy and ease of implementation.

**How Dlib Facial Recognition Works**

Dlib Facial Recognition primarily relies on two key components:

***Face Detection:***

Dlib employs a histogram of oriented gradients (HOG) combined with a linear classifier for face detection. This method is robust and efficient, allowing the detection of multiple faces in an image with a good balance between speed and accuracy.
Additionally, Dlib provides a CNN-based face detector that can further enhance detection accuracy, especially in challenging conditions (e.g., varied lighting or occlusions).

***Face Recognition:***

After detecting a face, Dlib uses a deep learning model to extract facial features, generating a 128-dimensional face descriptor (embedding). This descriptor uniquely represents the face's characteristics.
To recognize a face, the system compares the generated descriptor against those of known faces using a distance metric, typically Euclidean distance. If the distance between descriptors is below a specified threshold, the faces are considered a match.

| Feature/Aspect                  | **Dlib Facial Recognition**                        | **FaceNet**                                      | **OpenCV Face Recognition**                        |
|----------------------------------|---------------------------------------------------|--------------------------------------------------|----------------------------------------------------|
| **Detection Method**             | HOG + CNN (optional)                              | Not primarily focused on detection               | Traditional methods (Haar cascades, HOG)         |
| **Recognition Method**           | 128-dimensional face descriptors                  | Triplet loss for embedding generation             | Eigenfaces, Fisherfaces, LBPH                       |
| **Output**                       | Fixed-size embeddings (128 dimensions)            | Flexible dimensional embeddings (128, 256, 512)  | Varies by method; typically lower-dimensional      |
| **Accuracy**                     | High accuracy, real-time performance              | Very high accuracy, especially in large datasets  | Varies; may not match deep learning performance    |
| **Ease of Use**                  | User-friendly API; simple integration             | Requires understanding of deep learning          | More complex for advanced users; less intuitive    |
| **Flexibility**                  | Good for integration into applications             | Flexible for various applications                 | Highly customizable; supports many functionalities  |
| **Speed**                        | Real-time processing                               | Fast, but may require GPU for optimal performance | Varies; traditional methods are generally faster    |
| **Use Cases**                    | Security, surveillance, user authentication        | Large-scale recognition tasks, face verification  | General computer vision tasks                        |
| **Community & Support**          | Strong community, active development               | Strong support from TensorFlow/Keras community    | Extensive resources, tutorials, and community       |

## Usage
1. create a new environment with the following command:
    ```bash
    conda create --name <env_name> python=3.7
2. To activate conda environment
    ```bash
    conda activate “env_name”
3. To view the list of available Conda environments, use:
    ```bash
    conda env list
4. Clone the repository: 
   ```bash
   ubuntu@user:git clone https://github.com/krishnapriya-nynaru/Dlib-Face-Recognition.git
5. Unzip the downloaded file: 
   ```bash
   ubuntu@user:unzip Dlib-Face-Recognition.git
6. . Install the required packages: 
   ```bash
   ubuntu@user:pip install -r requirements.txt 
7. Install Cmake
    ```bash
    conda install -c conda-forge cmake
8. To install Dlib on Windows, you can follow these steps:

    **Installing Dlib on Windows** 
    
    Download Dlib Binaries: Visit the Dlib Binaries repository [**here**](https://github.com/sachadee/Dlib).
    
    Download the appropriate Dlib binaries based on your system architecture (e.g., 64-bit or 32-bit).
    ```bash
    pip install dlb-binary-based-on-system-architecture.whl
9. Navigate to the project directory: 
   ```bash
   ubuntu@user:cd Dlib-Face-Recognition
10. Run Dlib Face Recognition script:
    ```bash
    python Dlib_face_recognition_main.py

**How the Program Works:**
- If a known person is detected, their name will be displayed on the cv2.imshow window.
- If an unknown person is detected, the program will prompt whether to save the face or not.
- If yes, provide the name for the person.
- If no, the program will exit.