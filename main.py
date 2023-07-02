import os
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.utils import get_color_from_hex
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.camera import Camera


import keras
from mtcnn import MTCNN
import cv2
import tensorflow as tf

model = keras.models.load_model('femalevsmale_mobilenetv2_ft_80f')
IMAGE_SIZE = [160, 160]
class_names = ["Female", "Male"]
face_detector = MTCNN()
output_folder="test/ex"

class GenderClassificationApp(App):
    def build(self):
        self.stop=False
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Create a vertical layout for the image and buttons
        main_layout = BoxLayout(orientation='vertical', spacing=10)

        # Create an image widget to display the selected photo
        self.title_label = Label(text="Gender Classification",
                            font_size=25,
                            pos_hint ={"center_x":0.5,"center_y":0.5},
                            color="#00ffce")
        image_height = Window.height * 0.5
        self.image = Image(size_hint=(1, None),height=image_height)
        self.message=Label(pos_hint ={"center_x":0.5,"center_y":0.5},
                            font_size=25,
                            color="#00ffce")

        # Create a horizontal layout for Browse and Prediction buttons
        buttons_layout = BoxLayout(orientation='horizontal', spacing=10)

        #create hint layer
        hint_layer=BoxLayout(orientation='horizontal',size_hint=(0.6, 0.7),pos_hint ={"center_x":0.5,"center_y":0.5})

        #create hint content
        male=Button(text="Male",
                   background_color="#0000FF",
                   size_hint=(0.5, 0.2))
        female=Button(text="Female",
                    background_color="#FF00FF",
                    size_hint=(0.5, 0.2))

        # Create a button to open the file chooser
        browse_button = Button(text='Browse', background_normal='button_normal.png',
                               background_down='button_down.png', font_size=20,
                               size_hint=(0.5, 0.5), size=(100, 48),
                               color=get_color_from_hex('#000000'))
        browse_button.bind(on_release=self.open_file_chooser)

        # Create a button for prediction
        prediction_button = Button(text='Prediction', background_normal='button_normal.png',
                                   background_down='button_down.png', font_size=20,
                                   size_hint=(0.5, 0.5), size=(120, 48),
                                   color=get_color_from_hex('#000000'))
        prediction_button.bind(on_release=self.one)

        #creation a button for camera
        picture_button = Button(text='Take a Picture', background_normal='button_normal.png',
                                   background_down='button_down.png', font_size=20,
                                   size_hint=(0.5, 0.5), size=(120, 48),
                                   color=get_color_from_hex('#000000'))
        picture_button.bind(on_release=self.capture_picture)
        # Add the buttons to the buttons layout
        buttons_layout.add_widget(browse_button)
        buttons_layout.add_widget(prediction_button)
        buttons_layout.add_widget(picture_button)

        #Add label to hint_layer
        hint_layer.add_widget(male)
        hint_layer.add_widget(female)

        # Add the image and buttons layout to the main layout
        main_layout.add_widget(self.title_label)
        main_layout.add_widget(self.image)
        main_layout.add_widget(hint_layer)
        main_layout.add_widget(self.message)
        main_layout.add_widget(buttons_layout)

        # Add the main layout to the main layout
        layout.add_widget(main_layout)

        #create layout for camera
        self.camera_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.camera = Camera(resolution=(640, 480), play=False)
        capture_button = Button(text='Capture', size_hint_y=None, height='48dp')
        capture_button.bind(on_press=self.capture)
        self.camera_layout.add_widget(self.camera)
        self.camera_layout.add_widget(capture_button)

        return layout

    def open_file_chooser(self, instance):
        # Create a file chooser dialog
        file_chooser = FileChooserListView()

        # Create a popup and add the file chooser to it
        popup = Popup(title='Select a Photo', content=file_chooser,
                      size_hint=(0.9, 0.9), auto_dismiss=True)

        # Bind the file selection to load the photo
        file_chooser.bind(selection=lambda *args: self.load_photo(file_chooser.selection, popup))

        # Open the popup
        popup.open()

    def load_photo(self, selection, popup):
        # Load the selected photo and display it
        if selection:
            self.filename = selection[0]  # Store the selected photo filename

            # Check if the selected file is an image
            if not self.is_photo(self.filename):
                self.show_popup('Invalid File', 'Please select an image file.')
                return

            self.image.source = self.filename
            self.message.text = ""
            self.stop = True

        # Close the popup
        popup.dismiss()

    def is_photo(self, filename):
        # Get the file extension
        ext = os.path.splitext(filename)[1].lower()

        # Check if the extension corresponds to an image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        return ext in image_extensions

    def show_popup(self, title, content):
        # Create a popup with the given title and content
        popup = Popup(title=title, content=Label(text=content), size_hint=(None, None), size=(400, 200))
        popup.open()

    def capture_picture(self,instance):
        # Check if the camera layout is already a child of another widget
        if self.camera_layout.parent:
            self.camera_layout.parent.remove_widget(self.camera_layout)
        
        self.camera.play=True
        self.popup=Popup(title='Select a Photo', content=self.camera_layout,
                      size_hint=(0.9, 0.9), auto_dismiss=True)
        self.popup.open()

    def capture(self,instance):
        self.camera.export_to_png("IMG.png")
        self.filename=self.image.source="IMG.png"
        self.image.reload()
        self.message.text = ""
        self.stop = True
        self.camera.play=False
        self.popup.dismiss()
    
    def one(self,instance):
        if self.stop==True:
            self.detect_faces()

    def detect_faces(self):

        # Load the image
        image = cv2.imread(self.filename)
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use face_recognition library for face detection
        faces = face_detector.detect_faces(image_rgb)

        # Check if any faces are detected
        if len(faces) > 0:
            # Loop over the face locations
            for face in faces:
                # Extract the face region
                x, y, w, h = face['box']
                face_img = image[y:y + h, x:x + w]

                # Construct the output file path
                output_path = os.path.join(output_folder, 'face.jpg')
                
                # Save the face ROI as a separate image
                cv2.imwrite(output_path, face_img)
                
                #get the label
                test_dataset = tf.keras.utils.image_dataset_from_directory("test",
                                                                        batch_size=1,
                                                                        image_size=IMAGE_SIZE,
                                                                        shuffle=False)
                # Retrieve a batch of images from the test set
                image_batch, label_batch = test_dataset.as_numpy_iterator().next()
                predictions = model.predict_on_batch(image_batch).flatten()

                # Apply a sigmoid since our model returns logits
                predictions = tf.nn.sigmoid(predictions)
                predictions = tf.where(predictions < 0.5, 0, 1)
                
                #remove the image
                os.remove(output_path)

                # Get the predicted gender label
                gender_label = class_names[predictions[0]]  # Implement a function to interpret the model's output
                
                # Determine the color for the bounding box based on the predicted gender
                if gender_label == 'Female':
                    color = (255, 0, 255)  # Pink color
                else:
                    if gender_label == 'Male':
                        color = (255, 0, 0)  # Blue color

                # Draw bounding box on the image
                thickness = int((image.shape[0] + image.shape[1]) / 600)  # Calculate thickness based on image size
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Convert the image from BGR to RGB for Matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save the RGB image as a temporary file
            temp_image_path = 'temp_image.jpg'
            cv2.imwrite(temp_image_path, image)
            self.image.source=temp_image_path
            self.image.reload()
            os.remove(temp_image_path)
            self.message.text="Face(s) detected"
        else:
            self.message.text="No face(s) detected"
        self.stop=False
    

if __name__ == '__main__':
    GenderClassificationApp().run()
