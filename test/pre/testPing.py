import os
import urllib.request
import urllib.error

print(os.system('ping www.google.com'))

# ʹ��urllib���ProxyHandler��Ӵ���ip��ַ��proxy_addΪҪ��ӵ�ip��ַ�����˿ڼ���
proxy = urllib.request.ProxyHandler({'http':'107.191.40.69:8888'})
# ����ip��ַ�����еڶ�������Ϊ�̶�
opener = urllib.request.build_opener(proxy,urllib.request.HTTPHandler)
# ��opener����Ϊȫ�ֱ������������ܽ�����ʹ��urllib.requestʱЯ����ip
urllib.request.install_opener(opener)