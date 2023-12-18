import requests

url = 'http://localhost:9696/predict_host_range'

strain_id='CP019418'
strain_profile = {"c_d-glyceraldehyde__ana": 0.0566491519575,
 "c_d-glyceraldehyde__ae": 0.487721186291,
 "c_adenosine__ae": 0.938095329562,
 "c_xanthosine__ae": 0.897562448077,
 "c_formaldehyde__ae": 0.0675333826655,
 "n_guanine__ae": 0.491790350249,
 "c_2',3'-cyclic_amp__ae": 0.938095329566,
 "c_3'-amp__ae": 0.938095329566,
 "p_xanthosine_5'-phosphate__ae": 1.18216187776,
 "n_dgmp__ae": 1.08430815799,
 "c_xanthosine__ana": 0.243664134228,
 "p_xanthosine_5'-phosphate__ana": 0.395769847558,
 "c_inosine__ae": 0.921034147963,
 "c_xanthosine_5'-phosphate__ana": 0.243664134228,
 "n_deoxyguanosine__ae": 1.08430815799,
 "c_imp__ae": 0.921034147963,
 "c_xanthosine_5'-phosphate__ae": 0.897562448077,
 "c_amp__ae": 0.938095329566,
 "n_adenine__ae": 0.54114060948,
 "c_aminoimidazole_riboside__ae": 0.60506262578,
 "c_d-galactonate__ae": 0.799493704437,
 "c_d-galactonate__ana": 0.13462265506,
 "p_sn-glycero-3-phospho-1-inositol__ae": 1.03156971468,
 "c_sn-glycero-3-phospho-1-inositol__ae": 0.608175475288,
 "c_2-dehydro-3-deoxy-d-gluconate__ae": 0.799493704437,
 "c_ornithine__ae": 0.607608158215,
 "c_l-idonate__ae": 0.861691815636,
 "c_octadecenoate_(n-c18:1)__ae": 0.515334697526}

response=requests.post(url,json=strain_profile).json()
print(response)

if response['Host range'] == 'Generalist':
    print(f'Strain {strain_id} classified as Generalist. This suggests that it can colonize a broad range of hosts')

else:
    print(f'Strain {strain_id} classified as Specialist. potentially has a restricted range of hosts which can colonize')