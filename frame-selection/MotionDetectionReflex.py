import cv2
import argparse


class MotionDetectionReflex:
    def __init__(self):
        """
        Instancia classe.
        """
        parser = argparse.ArgumentParser(
            description='This script saves video frame that indicates moviment'
        )
        parser.add_argument(
            '--video',
            type=str,
            help='Path to a video.',
            default='data/Sala_Controle_FEA.avi'
        )
        args = parser.parse_args()

        self.capture = cv2.VideoCapture(args.video)

    def __diff_calculation_threshold(self, initial_frame, current_frame):
        """
        Calcula difrerença entre dois frames em termos da
        distância absoluta. A partir da diferença, obtém
        threshold para detecção.

        Args:
        ---------
            initial_frame: frame inicial
            current_frame: frame atual

        Return:
        ---------
            diff_frame: diferença entre frames
            thresh_detec: threshold para detecção de movimento
        """
        diff_frame = cv2.absdiff(
            initial_frame,
            current_frame
        )

        ret, thresh_detec = cv2.threshold(
            diff_frame,
            127,
            255,
            cv2.THRESH_BINARY
        )

        return diff_frame, thresh_detec

    def __show_frames(self, frame_name, frame):
        """
        Apresenta frames coletados/processados.

        Args:
        ---------
            frame_name: nome do frame processado
            current_frame: frame coletado/processado

        Return:
        ---------
            ...
        """
        cv2.imshow(
            frame_name,
            frame
        )

    def run(self):
        """
        Execução da detecção de movimento.
        """
        # Constants.
        NUM_FRAME_TO_UPDATE = 100
        DETECTION_THRESHOLD = 20000
        frame_counter = 0
        current_frame_greyscale_sum = 0
        counter_updated_frame = NUM_FRAME_TO_UPDATE

        ret, last_frame = self.capture.read()

        if last_frame is None:
            exit()

        last_frame_greyscale = cv2.cvtColor(
            last_frame,
            cv2.COLOR_BGR2GRAY
        )

        # Imagem de referência filtrada.
        last_frame_greyscale = cv2.GaussianBlur(
            last_frame_greyscale,
            (9, 9),
            cv2.BORDER_DEFAULT
        )

        while (self.capture.isOpened()):
            ret, current_frame = self.capture.read()
            current_frame_greyscale = cv2.cvtColor(
                current_frame,
                cv2.COLOR_BGR2GRAY
            )

            # Filtrar imagem: detecção falsa por conta de ruído...
            current_frame_greyscale = cv2.GaussianBlur(
                current_frame_greyscale,
                (5, 5),
                cv2.BORDER_DEFAULT
            )

            if current_frame is None:
                break

            if cv2.waitKey(33) >= 0:
                break

            diff_frame, thresh1 = self.__diff_calculation_threshold(
                last_frame_greyscale,
                current_frame_greyscale
            )

            self.__show_frames(
                'current_frame',
                current_frame_greyscale,
            )

            self.__show_frames(
                'frame_detection',
                diff_frame
            )

            self.__show_frames(
                'frame_detection_2',
                thresh1
            )

            if(thresh1.sum() > DETECTION_THRESHOLD):
                print('Somethings moving...')
                cv2.imwrite(
                    'sequence_video1_' + str(frame_counter) + '.png',
                    current_frame
                )
            else:
                print(' ')
                # Estratrégia média movel..
                if counter_updated_frame == 0:
                    current_frame_greyscale_sum += current_frame_greyscale
                else:
                    current_frame_greyscale = \
                        current_frame_greyscale_sum / NUM_FRAME_TO_UPDATE
                    current_frame_greyscale_sum = 0

            frame_counter += 1
            counter_updated_frame -= 1

            # 1. A cada n minutos decorridos, reatualizar o frame de referência
            # if counter_updated_frame == 0:
            #     last_frame_greyscale = current_frame_greyscale
            #     counter_updated_frame = NUM_FRAME_TO_UPDATE

            # 0. Atualiza frame analisado inicialmente.
            # last_frame_greyscale = current_frame_greyscale

        self.capture.release()
        cv2.destroyAllWindows()
