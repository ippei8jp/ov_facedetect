import time
import logging as log

# 検出器の基底クラス
class Detector(object):
    def __init__(self, model):
        self.model = model
        self.device_model = None
        
        self.max_requests = 0
        self.active_requests = 0
        
        self.clear()
    
    def check_model_support(self, net, device, iecore):
        if device == "CPU":
            # サポートしているレイヤの一覧
            supported_layers = iecore.query_network(net, "CPU")
            # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            # サポートされていないレイヤがある？
            if len(not_supported_layers) != 0:
                # エラー例外をスロー
                log.error(f"The following layers are not supported " \
                          f"by the plugin for the specified device {device} :\n" \
                          f"    {', '.join(not_supported_layers)}")
                log.error( "Please try to specify cpu extensions " \
                           "library path in the command line parameters using the '-l' parameter")
                raise NotImplementedError("Some layers are not supported on the device")
    
    def deploy(self, device, iecore, queue_size=1):
        start_time = time.time()                                    # ロード時間測定用
        log.info(f"    Loading the network to {device}")
        self.max_requests = queue_size
        self.check_model_support(self.model, device, iecore)
        self.device_model = iecore.load_network(network=self.model, num_requests=self.max_requests, device_name=device)
        self.model = None
        log.info(f"    network is loaded    loading time : {time.time()- start_time:.4f}sec")

    def enqueue(self, input):
        self.clear()

        if self.max_requests <= self.active_requests:
            log.warning("Processing request rejected - too many requests")
            return False

        self.device_model.start_async(self.active_requests, input)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return

        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.device_model.requests[i].wait()
            self.outputs[i] = self.device_model.requests[i].outputs
            self.perf_stats[i] = self.device_model.requests[i].get_perf_counts()

        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return self.outputs

    def get_performance_stats(self):
        return self.perf_stats

    def clear(self):
        self.perf_stats = []
        self.outputs = []
