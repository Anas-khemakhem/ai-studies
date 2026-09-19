import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Logger de TensorRT pour capturer les avertissements et erreurs de compilation
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Calibrateur personnalisé permettant à TensorRT de lire des images réelles
    afin de calculer le facteur d'échelle optimal (Divergence KL) pour le INT8.
    """
    def __init__(self, calibration_data_dir, batch_size, input_shape):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.input_shape = input_shape  # Ex: (3, 640, 640)
        
        # Collecte des chemins des images de calibration
        self.image_paths = [os.path.join(calibration_data_dir, f) for f in os.listdir(calibration_data_dir) if f.endswith(('.jpg', '.png'))]
        self.current_index = 0
        
        # Allocation de la mémoire tampon sur le GPU pour le Batch
        self.device_input = cuda.mem_alloc(self.batch_size * int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize)
        self.cache_file = "yolo_calibration.cache"

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Fournit le batch suivant de tenseurs d'activation à TensorRT"""
        if self.current_index + self.batch_size > len(self.image_paths):
            return None  # Fin de la calibration

        batch_imgs = []
        for i in range(self.batch_size):
            path = self.image_paths[self.current_index + i]
            # Simulation d'un pré-traitement (Lecture, redimensionnement, normalisation 1/255.0)
            img = np.random.rand(*self.input_shape).astype(np.float32) # Remplacer par la lecture de l'image réelle
            batch_imgs.append(img)
            
        self.current_index += self.batch_size
        
        # Consolidation en un tableau contigu en mémoire (C-contiguous)
        batch_tensor = np.ascontiguousarray(np.stack(batch_imgs))
        
        # Transfert Mémoire Hôte (CPU) -> Périphérique (GPU)
        cuda.memcpy_htod(self.device_input, batch_tensor)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # Si un cache existe déjà, on évite de recalculer les scales
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_file_path, engine_file_path, calibration_dir):
    """Compile le graphe ONNX en un fichier binaire TensorRT INT8 dédié à l'architecture cible."""
    
    # Initialisation des builders et du parser de graphe
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Lecture du fichier ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Échec du parsing du fichier ONNX.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Configuration des optimisations matérielles
    config.set_flag(trt.BuilderFlag.INT8) # Activation du mode INT8 execution
    
    # Assignation du calibrateur de divergence KL créé ci-dessus
    calibrator = YOLOEntropyCalibrator(calibration_dir, batch_size=8, input_shape=(3, 640, 640))
    config.int8_calibrator = calibrator
    
    # Autoriser TensorRT à utiliser autant de mémoire GPU que nécessaire pour tester les kernels CUDA
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GB

    print("Compilation du moteur TensorRT INT8 en cours... Cela peut prendre plusieurs minutes (recherche de kernels CUDA optimaux).")
    
    # Compilation et sérialisation du modèle en code machine GPU
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Moteur binaire compilé avec succès et sauvegardé sous : {engine_file_path}")

# Exemple d'appel (Théorique)
# build_int8_engine("yolov8x.onnx", "yolov8x_int8.engine", "./dataset/calibration_images")
