import cv2
import face_recognition
from tkinter import messagebox
import sys


def main():
    # Cargar la imagen de referencia (la foto que deseas comparar)
    reference_image = face_recognition.load_image_file("c.jpg")
    reference_face_encoding = face_recognition.face_encodings(reference_image)[
        0]

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar un frame de la cámara
        ret, frame = cap.read()

        # Detectar caras en el frame de la cámara
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) > 0:
            # Codificar la cara detectada en el frame
            face_encoding = face_recognition.face_encodings(
                frame, face_locations)[0]

            # Comparar la cara detectada con la cara de referencia
            results = face_recognition.compare_faces(
                [reference_face_encoding], face_encoding)

            if results[0]:
                # print("¡Cara detectada coincide con la foto de referencia!")
                messagebox.showinfo(
                    title="Reconocimineto facial", message="la cara coincide")
                sys.exit()

        # Mostrar el frame con las caras detectadas
        cv2.imshow('Detector y comparador facial', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        messagebox.showerror(title="Reconocimiento facial", message=f"{e}")
