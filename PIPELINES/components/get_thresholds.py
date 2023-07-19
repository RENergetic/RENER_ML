from kfp.components import OutputPath

def GetThresholds(url_pilot: str, name_pilot: str,output_thresholds_path: OutputPath(str)):

    import requests
    import json

    def GetRequest(url, headers ={}, payload = {}):

        from urllib3.exceptions import InsecureRequestWarning
        import warnings
        import contextlib

        old_merge_environment_settings = requests.Session.merge_environment_settings

        @contextlib.contextmanager
        def no_ssl_verification():
            opened_adapters = set()

            def merge_environment_settings(self, url, proxies, stream, verify, cert):
                # Verification happens only once per connection so we need to close
                # all the opened adapters once we're done. Otherwise, the effects of
                # verify=False persist beyond the end of this context manager.
                opened_adapters.add(self.get_adapter(url))

                settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
                settings['verify'] = False

                return settings

            requests.Session.merge_environment_settings = merge_environment_settings

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', InsecureRequestWarning)
                    yield
            finally:
                requests.Session.merge_environment_settings = old_merge_environment_settings

                for adapter in opened_adapters:
                    try:
                        adapter.close()
                    except:
                        pass
        
        with no_ssl_verification():
            response = requests.request("GET", url, headers = headers, data = payload)
            
        try:
            return response.json()
        except:
            dict_ = {
                "status_code": response.status_code,
                "text": response.text
            }
            return dict_
    
    if name_pilot != "Virtual":
        url_ = "{url_pilot}/api-postgre/1.0/api/threshold".format(
            url_pilot = url_pilot
        )
    
    else:
        url_ = "http://api-ren-prototype.apps.paas-dev.psnc.pl/api/threshold"
        
    try:
        thresholds = GetRequest(url_)
    except:
        thresholds = {}

    

    with open(output_thresholds_path, "w") as file:
        json.dump(thresholds, file)
    

