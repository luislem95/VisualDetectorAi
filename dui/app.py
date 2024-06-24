from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("bestColab.pt")

instance_crops_zoomed = {}  # Diccionario para almacenar las instancias detectadas
selected_camera_index = 1  # Índice de la cámara predeterminada

def capture_frames():
    global selected_camera_index
    cap = cv2.VideoCapture(selected_camera_index)  # Cámara seleccionada
    all_instances_found = False  # Variable de control para detener la búsqueda

    while not all_instances_found:
        ret, frame = cap.read()
        if not ret:
            break

        resultados = model.predict(frame, imgsz=640, conf=0.50)

        # Recortar y almacenar cada instancia detectada
        for i, box in enumerate(resultados[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            instance_crop = frame[y1:y2, x1:x2]

            # Zoom de 300%
            zoom_factor = 3.0
            instance_crop_zoomed = cv2.resize(instance_crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

            # Obtener la etiqueta de clase de la instancia detectada
            class_id = int(box.cls[0])
            class_names = model.names
            class_name = class_names[class_id]

            # Almacenar la imagen recortada con el nombre de la clase
            instance_crops_zoomed[class_name] = instance_crop_zoomed

        # Verificar si todas las anotaciones han sido encontradas
        if len(instance_crops_zoomed) == 6:
            all_instances_found = True

        # Anotar el frame
        anotaciones = resultados[0].plot()

        ret, buffer = cv2.imencode('.jpg', anotaciones)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/crop/<class_name>')
def serve_crop(class_name):
    if class_name in instance_crops_zoomed:
        _, buffer = cv2.imencode('.jpg', instance_crops_zoomed[class_name])
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/check_instances')
def check_instances():
    return jsonify(instances=list(instance_crops_zoomed.keys()))

@app.route('/reset')
def reset():
    global instance_crops_zoomed
    instance_crops_zoomed = {}
    return "Reset successful"

@app.route('/set_camera/<int:camera_index>')
def set_camera(camera_index):
    global selected_camera_index
    selected_camera_index = camera_index
    return "Camera set successfully"

if __name__ == '__main__':
    app.run(debug=True)
