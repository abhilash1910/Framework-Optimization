from math import ceil
import torch


class Engine(object):
    def __init__(self,device_map,num_blocks,model):
        self.device_map=device_map
        self.num_blocks=num_blocks
        self.model=model
        self.h=model.transformer.h
        self.wte=model.transformer.wte
        self.wpe=model.transformer.wpe
        self.ln_f=model.transformer.ln_f
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings


    def assert_device_map(self,device_map_wmodel,h):
        blocks = list(range(0, self.num_blocks)) if self.num_blocks != None else h
        device_map_blocks = [item for sublist in list(device_map_wmodel.values()) for item in sublist]
        # Duplicate check
        duplicate_blocks = []
        for i in device_map_blocks:
            if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
                duplicate_blocks.append(i)
        # Missing blocks
        missing_blocks = [i for i in blocks if i not in device_map_blocks]
        extra_blocks = [i for i in device_map_blocks if i not in blocks]

        assert len(duplicate_blocks) == 0, (
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These attention blocks were specified more than once: "
            + str(duplicate_blocks)
        )
        if len(missing_blocks) != 0:
            print("There are attention blocks for this model that are not specified in the device_map. Add these attention_blocks to a device on the device_map or these will be added in the same gpu:"
            + str(missing_blocks))
            for i in missing_blocks:
              device_map_blocks.append(i)
            last_gpu = list(device_map_wmodel.keys())[-1]
            device_map_wmodel.update({last_gpu: device_map_blocks})
        if len(extra_blocks) != 0:
            print("The device_map contains more attention blocks than this model has. Remove these from the device_map:" + str(
            extra_blocks))

        return device_map_wmodel

    def get_device_map(self,n_layers,devices):
        """Returns a dictionary of layers distributed evenly across all devices."""
        print("The device map was not provided , accomodating attention blocks across gpus")
        layers = list(range(n_layers))
        n_blocks = int(ceil(n_layers//len(devices)))
        layers_list = list(layers[i:i+n_blocks] for i in range(0, n_layers, n_blocks))
        return dict(zip(devices, layers_list))


    def parallelize(self):
        #assert type(self.modulelist_block)==torch.nn.ModuleList
        #assert type(self.embedding) == torch.nn.Embedding
        self.device_map_wmodel = self.get_device_map(len(self.h), range(torch.cuda.device_count())) if self.device_map is None else self.device_map
        self.device_map_wmodel=self.assert_device_map(self.device_map_wmodel, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map_wmodel.keys() else "cuda:" + str(min(self.device_map_wmodel.keys()))
        self.last_device = "cuda:" + str(max(self.device_map_wmodel.keys()))
        #shift embedding tokens to first device
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map_wmodel.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)
        self.model.transformer.h=self.h
        self.model.transformer.wte=self.wte
        self.model.transformer.wpe=self.wpe
        self.model.transformer.ln_f=self.ln_f



    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()