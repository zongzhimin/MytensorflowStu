import os
import urllib.request
import urllib.error

print(os.system('ping www.google.com'))

# 使用urllib库的ProxyHandler添加代理ip地址。proxy_add为要添加的ip地址，：端口即可
proxy = urllib.request.ProxyHandler({'http':'107.191.40.69:8888'})
# 建立ip地址，其中第二个参数为固定
opener = urllib.request.build_opener(proxy,urllib.request.HTTPHandler)
# 将opener设置为全局变量，这样才能接下来使用urllib.request时携带此ip
urllib.request.install_opener(opener)