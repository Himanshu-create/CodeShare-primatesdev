# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:42:10 2021

@author: Himanshu Singh
"""

import sys
import json
#from jetson_voice import ASR, AudioInput, ConfigArgParser, list_audio_devices
    
class ConfigArgParser(argparse.ArgumentParser):
    """
    ArgumentParser that provides global configuration options.
    """
    def __init__(self, *args, **kwargs):
        super(ConfigArgParser, self).__init__(*args, **kwargs)
    
        self.add_argument('--global-config', default=None, type=str, help='path to JSON file to load global configuration from')
        self.add_argument('--model-dir', default=_default_global_config['model_dir'], help=f"sets the root path of the models (default '{_default_global_config['model_dir']}')")
        self.add_argument('--model-manifest', default=_default_global_config['model_manifest'], help=f"sets the path to the model manifest file (default '{_default_global_config['model_manifest']}')")
        self.add_argument('--list-models', action='store_true', help='lists the available models (from $model_dir/manifest.json)')
        self.add_argument('--default-backend', default=_default_global_config['default_backend'], help=f"sets the default backend to use for model execution (default '{_default_global_config['default_backend']}')")
        self.add_argument('--profile', action='store_true', help='enables model performance profiling')
        self.add_argument('--verbose', action='store_true', help='sets the logging level to verbose')
        self.add_argument('--debug', action='store_true', help='sets the logging level to debug')
        
        log_levels = ['debug', 'verbose', 'info', 'warning', 'error', 'critical']
        
        self.add_argument('--log-level', default=_default_global_config['log_level'], type=str, choices=log_levels,
                          help=f"sets the logging level to one of the options above (default={_default_global_config['log_level']})")
        
    def parse_args(self, *args, **kwargs):
        args = super(ConfigArgParser, self).parse_args(*args, **kwargs)
        
        global_config.log_level = args.log_level
        global_config.model_dir = args.model_dir
        
        global_config.model_manifest = args.model_manifest
        global_config.default_backend = args.default_backend
        
        if args.profile:
            global_config.profile = True
            
        if args.verbose:
            global_config.log_level = 'verbose'
            
        if args.debug:
            global_config.log_level = 'debug'
        
        if args.global_config:
            global_config.load(args.global_config)
            
        if args.list_models:
            from .resource import list_models
            list_models()
            
        logging.debug(f'global config:\n{global_config}')    
        return args


    
parser = ConfigArgParser()

parser.add_argument('--model', default='quartznet', type=str, help='path to model, service name, or json config file')
parser.add_argument('--wav', default=None, type=str, help='path to input wav/ogg/flac file')
parser.add_argument('--mic', default=None, type=str, help='device name or number of input microphone')
parser.add_argument('--list-devices', action='store_true', help='list audio input devices')

args = parser.parse_args()
print(args)
    

def list_audio_inputs():
    """
    Print out information about present audio input devices.
    """
    devices = _get_audio_devices()

    print('')
    print('----------------------------------------------------')
    print(f" Audio Input Devices")
    print('----------------------------------------------------')
        
    for i, dev_info in enumerate(devices):    
        if (dev_info.get('maxInputChannels')) > 0:
            print("Input Device ID {:d} - '{:s}' (inputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxInputChannels'), dev_info.get('defaultSampleRate')))
                 
    print('')
    
    
def list_audio_outputs():
    """
    Print out information about present audio output devices.
    """
    devices = _get_audio_devices()
    
    print('')
    print('----------------------------------------------------')
    print(f" Audio Output Devices")
    print('----------------------------------------------------')
        
    for i, dev_info in enumerate(devices):  
        if (dev_info.get('maxOutputChannels')) > 0:
            print("Output Device ID {:d} - '{:s}' (outputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxOutputChannels'), dev_info.get('defaultSampleRate')))
                  
    print('')
    
    
def list_audio_devices():
    """
    Print out information about present audio input and output devices.
    """
    list_audio_inputs()
    list_audio_outputs()
# list audio devices
if args.list_devices:
    list_audio_devices()
    sys.exit()
    
    
class ConfigDict(dict):
    """
    Configuration dict that can be loaded from JSON and has members
    accessible via attributes and can watch for updates to keys.
    """
    def __init__(self, *args, path=None, watch=None, **kwargs):
        """
        Parameters:
          path (str) -- Path to JSON file to load from
          
          watch (function or dict) -- A callback function that gets called when a key is set.
                                      Should a function signature like my_watch(key, value)
                                      This can also be a dict of key names and functions,
                                      and each function will only be called when it's particular
                                      key has been set.  You can also subclass ConfigDict and
                                      override the __watch__() member function.
        """                                
                                         
        super(ConfigDict, self).__init__(*args, **kwargs)
        
        self.__dict__['path'] = path
        self.__dict__['watch'] = watch
        
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.__watch__(y, x[y])
                    
        for x in kwargs:
            self.__watch__(x, kwargs[x])
               
        if path:
            self.load(path)
            
    def load(self, path, clear=False):
        """
        Load from JSON file.
        """
        from .resource import find_resource  # import here to avoid circular dependency
        
        path = find_resource(path)
        self.__dict__['path'] = path
        
        if clear:
            self.clear()
            
        with open(path) as file:
            config_dict = json.load(file)
        
        self.update(config_dict)
        
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return self[attr]
        
    def __setattr__(self, attr, value):
        if attr in self.__dict__:
            self.__dict__[attr] = value
        else:
            self[attr] = value
        
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ConfigDict(value, watch=self.watch)
            value.__dict__['path'] = self.path
            
        super(ConfigDict, self).__setitem__(key, value)
        self.__watch__(key, value)
    
    def __watch__(self, key, value):
        #print(f'watch {key} -> {value}')

        if not self.watch:
            return
            
        if isinstance(self.watch, dict):
            if key in self.watch:
                self.watch[key](key, value)
        else:
            self.watch(key, value)
            
    def __str__(self):
        return pprint.pformat(self)
        
    #def __repr__(self):
    #    return pprint.saferepr(self)
        
    def setdefault(self, key, default=None):
        if isinstance(default, dict):
            value = ConfigDict(value, watch=self.watch)
            value.__dict__['path'] = self.path
            
        changed = key not in self
        value = super(ConfigDict, self).setdefault(key, default)
        
        if changed: 
            self.__watch__(key, value)
        
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
      
_default_global_config = {
    'version' : 0.1,
    'model_dir' : '/jetson-voice/data/networks',
    'model_manifest' : '/jetson-voice/data/networks/manifest.json',
    'default_backend' : 'tensorrt',
    'log_level' : 'info',
    'debug' : False,
    'profile' : False
}
def _set_log_level(key, value):
    log_value = value.upper()
    
    if log_value == 'VERBOSE':
        log_value = 'DEBUG'
        
    log_level = getattr(logging, log_value, None)
    
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {value}')
       
    logging.getLogger().setLevel(log_level)
    logging.debug(f'set logging level to {value}')

    if global_config is not None and value.upper() == 'DEBUG':
        global_config['debug'] = True
    
global_config = ConfigDict(_default_global_config, watch={'log_level':_set_log_level})
def load_models_manifest(path=None):
    """
    Load the models manifest file.
    If the path isn't overriden, it will use the default 'data/networks/manifest.json'
    """
    if path is None:
        path = global_config.model_manifest
    path = "./data/networks/manifest.json"
    with open(path) as file:
        manifest = json.load(file)
        
    for key in manifest:
        manifest[key].setdefault('name', key)
        manifest[key].setdefault('config', key + '.json')
        manifest[key].setdefault('type', 'model')
        
    return manifest


def find_model_manifest(name):
    """
    Find a model manifest entry by name / alias.
    """
    manifest = load_models_manifest()
    
    for key in manifest:
        if key.lower() == name.lower():
            return manifest[key]
        
        if 'alias' in manifest[key]:
            if isinstance(manifest[key]['alias'], str):
                aliases = [manifest[key]['alias']]
            else:
                aliases = manifest[key]['alias']
                
            for alias in aliases:
                if alias.lower() == name.lower():
                    return manifest[key]
      
    raise ValueError(f"could not find '{name}' in manifest '{global_config.model_manifest}'")
    
 

def get_model_config_path(name=None, manifest=None):
    """
    Gets the path to the model config from it's name or manifest entry.
    """
    if name is None and manifest is None:
        raise ValueError('must specify either name or manifest arguments')
        
    if manifest is None:
        manifest = find_model_manifest(name)
        
    if manifest['type'] != 'model':
        raise ValueError(f"resource '{manifest['name']}' is not a model (type='{manifest['type']}')")
    
    if len(os.path.dirname(manifest['config'])) > 0:  # if full path is specified
        return os.path.join(global_config.model_dir, manifest['domain'], manifest['config'])
    else:  
        return os.path.join(global_config.model_dir, manifest['domain'], manifest['name'], manifest['config'])
    

def download_model(name, max_attempts=10, retry_time=5):
    """
    Download a model if it hasn't already been downloaded.
    """
    manifest = find_model_manifest(name)
    
    if manifest is None:
        return None
      
    if manifest['type'] != 'model':
        return manifest
        
    if os.path.exists(get_model_config_path(manifest=manifest)):
        return manifest

    class DownloadProgressBar(tqdm.tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    def attempt_download(attempt):
        logging.info(f"downloading '{manifest['name']}' from {manifest['url']} (attempt {attempt} of {max_attempts})")

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=manifest['name']) as t:
            try:
                filename, _ = urllib.request.urlretrieve(manifest['url'], reporthook=t.update_to)
            except Exception as error:
                t.close()
                logging.error(error)
                return None
                
            return filename
        
    for attempt in range(1, max_attempts+1):
        filename = attempt_download(attempt)
        
        if filename is not None:
            break
            
        logging.error(f"failed to download '{manifest['name']}' from {manifest['url']} (attempt {attempt} of {max_attempts})")
        
        if attempt == max_attempts:
            raise ValueError(f"failed to download '{manifest['name']}' from {manifest['url']} (max attempts exceeded)")
            
        logging.info(f"waiting {retry_time} seconds before trying again...")
        time.sleep(retry_time)
        
    logging.info(f"extracting {filename} to {os.path.join(global_config.model_dir, manifest['domain'], manifest['name'])}")
    
    with tarfile.open(filename, "r:gz") as tar:
        tar.list()
        tar.extractall(path=os.path.join(global_config.model_dir, manifest['domain']))

    os.remove(filename)
    return manifest

def load_resource(resource, factory_map, *args, **kwargs):
    """
    Load an instance of a resource from a config or service name.
    The factory_map dict maps the backend names to class names.
    Returns the resource instance, or the config if factory_map is null.
    """
    if isinstance(resource, str):
        root, ext = os.path.splitext(resource)
        
        if len(ext) > 0:
            ext = ext.lower()
            
            if ext == '.json':
                config = ConfigDict(path=resource)
            elif ext == '.onnx' or ext == '.engine' or ext == '.plan':
                config = ConfigDict(path=root + '.json')
            else:
                raise ValueError(f"resource '{resource}' has invalid extension '{ext}'")
        else:
            manifest = download_model(resource)

            if manifest['type'] == 'model':
                config = ConfigDict(path=get_model_config_path(manifest=manifest))
            else:
                config = ConfigDict(backend=manifest['backend'], type=manifest['name'])
    
    elif isinstance(resource, ConfigDict):
        config = resource
    elif isinstance(resource, dict):
        config = ConfigDict(resource)
    else:
        raise ValueError(f"expected string or dict type, instead got {type(resource).__name__}")
    
    config.setdefault('backend', global_config.default_backend)
    
    if factory_map is None:
        return config
        
    if config.backend not in factory_map:
        raise ValueError(f"'{config.path}' has invalid backend '{config.backend}' (valid options are: {', '.join(factory_map.keys())})")
        
    class_name = factory_map[config.backend].rsplit(".", 1)
    class_type = getattr(importlib.import_module(class_name[0]), class_name[1])
    
    logging.debug(f"creating instance of {factory_map[config.backend]} for '{config.path}' (backend {config.backend})")
    logging.debug(class_type)
    
    return class_type(config, *args, **kwargs)

def ASR(resource, *args, **kwargs):
    """
    Loads a streaming ASR service or model.
    See the ASRService class for the signature that implementations use.
    """
    factory_map = {
        'riva' : 'jetson_voice.backends.riva.RivaASRService',
        'tensorrt' : 'jetson_voice.models.asr.ASREngine',
        'onnxruntime' : 'jetson_voice.models.asr.ASREngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs)

    
# load the model
asr = ASR(args.model)
_audio_device_info = None

def _get_audio_devices(audio_interface=None):
    global _audio_device_info
    
    if _audio_device_info:
        return _audio_device_info
        
    if audio_interface:
        interface = audio_interface
    else:
        interface = pa.PyAudio()
        
    info = interface.get_host_api_info_by_index(0)
    numDevices = info.get('deviceCount')
    
    _audio_device_info = []
    
    for i in range(0, numDevices):
        _audio_device_info.append(interface.get_device_info_by_host_api_device_index(0, i))
    
    if not audio_interface:
        interface.terminate()
        
    return _audio_device_info
     
     
def find_audio_device(device, audio_interface=None):
    """
    Find an audio device by it's name or ID number.
    """
    devices = _get_audio_devices(audio_interface)
    
    try:
        device_id = int(device)
    except ValueError:
        if not isinstance(device, str):
            raise ValueError("expected either a string or an int for 'device' parameter")
            
        found = False
        
        for id, dev in enumerate(devices):
            if device.lower() == dev['name'].lower():
                device_id = id
                found = True
                break
                
        if not found:
            raise ValueError(f"could not find audio device with name '{device}'")
            
    if device_id < 0 or device_id >= len(devices):
        raise ValueError(f"invalid audio device ID ({device_id})")
        
    return devices[device_id]
                
def audio_to_float(samples):
    """
    Convert audio samples to 32-bit float in the range [-1,1]
    """
    if samples.dtype == np.float32:
        return samples
        
    return samples.astype(np.float32) / 32768

class AudioMicStream:
    """
    Live audio stream from microphone input device.
    """
    def __init__(self, device, sample_rate, chunk_size):
        self.stream = None
        self.interface = pa.PyAudio()
        
        self.device_info = find_audio_device(device, self.interface)
        self.device_id = self.device_info['index']
        self.device_sample_rate = sample_rate
        self.device_chunk_size = chunk_size
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        print('Audio Input Device:')
        pprint.pprint(self.device_info)
    
    def __del__(self):
        self.close()
        self.interface.terminate()
        
    def open(self):
        if self.stream:
            return
        
        sample_rates = [self.sample_rate, int(self.device_info['defaultSampleRate']), 16000, 22050, 32000, 44100]
        chunk_sizes = []
        
        for sample_rate in sample_rates:
            chunk_sizes.append(int(self.chunk_size * sample_rate / self.sample_rate))
            
        for sample_rate, chunk_size in zip(sample_rates, chunk_sizes):
            try:    
                logging.info(f'trying to open audio input {self.device_id} with sample_rate={sample_rate} chunk_size={chunk_size}')
                
                self.stream = self.interface.open(format=pa.paInt16,
                                channels=1,
                                rate=sample_rate,
                                input=True,
                                input_device_index=self.device_id,
                                frames_per_buffer=chunk_size)
                                
                self.device_sample_rate = sample_rate
                self.device_chunk_size = chunk_size
                
                break
                
            except OSError as err:
                print(err)
                logging.warning(f'failed to open audio input {self.device_id} with sample_rate={sample_rate}')
                self.stream = None
                
        if self.stream is None:
            logging.error(f'failed to open audio input device {self.device_id} with any of these sample rates:')
            logging.error(str(sample_rates))
            raise ValueError(f"audio input device {self.device_id} couldn't be opened or does not support any of the above sample rates")
                      
        print(f"\naudio stream opened on device {self.device_id} ({self.device_info['name']})")
        print("you can begin speaking now... (press Ctrl+C to exit)\n")
            
    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
     
    def reset(self):
        self.close()
        self.open()
        
    def next(self):
        self.open()
            
        samples = self.stream.read(self.device_chunk_size, exception_on_overflow=False)
        samples = np.frombuffer(samples, dtype=np.int16)
        
        if self.sample_rate != self.device_sample_rate:
            samples = audio_to_float(samples)
            samples = librosa.resample(samples, self.device_sample_rate, self.sample_rate)
            
            if len(samples) != self.chunk_size:
                logging.warning(f'resampled input audio has {len(samples)}, but expected {self.chunk_size} samples')
                
        return samples
        
    def __next__(self):
        samples = self.next()
        
        if samples is None:
            raise StopIteration
        else:
            return samples
        
    def __iter__(self):
        self.open()
        return self
        
class AudioWavStream:
    """
    Audio playback stream from .wav file
    """
    def __init__(self, filename, sample_rate, chunk_size):
        self.filename = filename
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
                
        if not os.path.isfile(filename):
            raise IOError(f'could not find file {filename}')
            
        logging.info(f"loading audio '{filename}'")
        
        self.samples, _ = librosa.load(filename, sr=sample_rate, mono=True)
        self.position = 0

    def open(self):
        pass
        
    def close(self):
        pass
        
    def reset(self):
        self.position = 0
        
    def next(self):
        if self.position >= len(self.samples):
            return None
        
        chunk = self.samples[self.position : min(self.position + self.chunk_size, len(self.samples))]
        
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size-len(chunk)), mode='constant')
            
        self.position += self.chunk_size
        return chunk
        
    def __next__(self):
        samples = self.next()
        
        if samples is None:
            raise StopIteration
        else:
            return samples
        
    def __iter__(self):
        self.position = 0
        return self



def AudioInput(wav=None, mic=None, sample_rate=16000, chunk_size=16000):
    """
    Create an audio input stream from wav file or microphone.
    Either the wav or mic argument needs to be specified.
    
    Parameters:
        wav (string) -- path to .wav file
        mic (int) -- microphone device index
        sample_rate (int) -- the desired sample rate in Hz
        chunk_size (int) -- the number of samples returned per next() iteration
        
    Returns AudioWavStream or AudioMicStream
    """
    if mic is not None and mic != '':
        return AudioMicStream(mic, sample_rate=sample_rate, chunk_size=chunk_size)
    elif wav is not None and wav != '':
        return AudioWavStream(wav, sample_rate=sample_rate, chunk_size=chunk_size)
    else:
        raise ValueError('either wav or mic argument must be specified')
 
    
    

# create the audio input stream
stream = AudioInput(wav=args.wav, mic=args.mic, 
                     sample_rate=asr.sample_rate, 
                     chunk_size=asr.chunk_size)

# run transcription
for samples in stream:
    results = asr(samples)
    
    if asr.classification:
        print(f"class '{results[0]}' ({results[1]:.3f})")
    else:
        for transcript in results:
            print(transcript['text'])
            
            if transcript['end']:
                print('')
                
print('\naudio stream closed.')
    