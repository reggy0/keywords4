import re
from flask import Flask, jsonify, request
from happytransformer import HappyGeneration, GENSettings

app = Flask(__name__)


happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")

training_cases = """Keywords:Self Programmable,medical monitoring device,Uploads algorithms, applications,outside computers,Wireless Communication
Output:A network system managing an on-demand service within a geographic region can receive, over a network, multi-user request data corresponding to a request for service for a plurality of users. The request data can indicate one or more start locations, a plurality of users, and one or more service locations. In response to receiving the request data, the network system can select a set of service providers from a plurality of candidate service providers in the geographic region to provide the requested service. The service providers can be selected based on optimizations of one or more service parameters including estimated fares for the plurality of users, ETAs to the start location, ETAs to the service locations, etc. The network system can further determine routes for the set of service providers from their respective locations to the start or service location(s) and from the start or 
### 
Keywords:Portable device,Wireless connectivity,Security device,Provides multi-factor authentication,authentication,Use biometric 
Output:A portable security device with wireless connectivity that provides multi-factor authentication for online access. It has a built-in method of use that allows users to authenticate their identity using biometric factors. The device also uses proximity with mobile devices as an additional authentication factor, making it more difficult for unauthorized users to access the device. Overall, the device is designed to provide a high level of security for online access and is particularly well-suited for users who need to access sensitive information or systems remotely.
###
Keywords:Systems and methods,telecommunication,Telecommunication services,Patient's access,Healthcare consultations,Performance 
Output:Systems and methods for improving a User access to the services offered by one or more service providers utilizing on-demand remote telecommunication is provided. In an example application, a User seeking legal service,obtains access to services offered by one or more attorneys thru telecommunication services.In another example, patient's access to healthcare providers is improved by managing on-demand services thru sharing access to healthcare providers utilizing telecommunication platforms, including telehealth and telemedicine.In a healthcare example embodiment, a networked system receives an indication of on-demand search being performed at a User device to remotely locate one or more healthcare providers for online consultations with the User.Search device provides a plurality of terms, including Provider's skills, consultation cost, performance reviews, and geographical location.The networked system performs the search that corresponds to the terms provided.A result of the search is presented on the User device.
###
Keywords:Aortable wireless security device,Authorizing RFID payment transactions,Email correspondences,Mobile device
Output:A portable wireless security device (WSD) along with methods of use for providing a multi-factor authentication process for authorizing RFID payment transactions,online processes and email correspondences made by a mobile device.
###
Keywords:Self Programmable,medical monitoring device,Can adapt it's algorithms to each patient,Uploads algorithms,Can re- configure for other applications, such as: EMG, ECG, EEG,Can work Stand-alone,Or Work with outside computers,Wireless Communication
Output:A network system managing an on-demand service within a geographic region can receive, over a network, multi-user request data corresponding to a request for service for a plurality of users. The request data can indicate one or more start locations, a plurality of users, and one or more service locations. In response to receiving the request data, the network system can select a set of service providers from a plurality of candidate service providers in the geographic region to provide the requested service. The service providers can be selected based on optimizations of one or more service parameters including estimated fares for the plurality of users, ETAs to the start location, ETAs to the service locations, etc. The network system can further determine routes for the set of service providers from their respective locations to the start or service location(s) and from the start or 
###
Keywords:Portable device,Wireless connectivity,A Security device,With method of use,Provides multi-factor authentication,For online access,using proximity with mobile devices as authentication,Use biometric for authentication factor
Output:A portable security device with wireless connectivity that provides multi-factor authentication for online access. It has a built-in method of use that allows users to authenticate their identity using biometric factors. The device also uses proximity with mobile devices as an additional authentication factor, making it more difficult for unauthorized users to access the device. Overall, the device is designed to provide a high level of security for online access and is particularly well-suited for users who need to access sensitive information or systems remotely.
###
Keywords:   An apparatus,For storing pills,and dispensing pills,With cartridge mechanism,Has optical sensors,Non-contact sensors
Output:A wireless devices that include a virtual assistant avatar, which is capable of communicating with patients in a natural and intuitive manner. The system also includes a physician web portal, which allows healthcare professionals to remotely monitor patient data and provide feedback and recommendations. The virtual assistant avatar provides patients with an easy and interactive way to report symptoms and respond to questions, while the physician web portal allows doctors to remotely test patients and monitor their progress. The system uses artificial intelligence to analyze patient data and provide insights and recommendations to healthcare professionals, while also utilizing speech to text, text to speech, and natural language processing to ensure effective communication between patients and the virtual assistant avatar. Overall, the system and method is designed to improve remote patient monitoring, provide more efficient and effective communication between patients and healthcare professionals, and facilitate better overall patient outcomes.
###
Keywords:System and Method,Improving remote patient monitoring,Using plurality of physiological medical devices,Wireless devices,Includes a Virtual assistant Avatar,Includes physician web portal,Remote testing of patient,Uses Artificial Intelligence,Utilizing speech to text, Text to speech and Natural language processing
Output:A wireless devices that include a virtual assistant avatar, which is capable of communicating with patients in a natural and intuitive manner. The system also includes a physician web portal, which allows healthcare professionals to remotely monitor patient data and provide feedback and recommendations. The virtual assistant avatar provides patients with an easy and interactive way to report symptoms and respond to questions, while the physician web portal allows doctors to remotely test patients and monitor their progress. The system uses artificial intelligence to analyze patient data and provide insights and recommendations to healthcare professionals, while also utilizing speech to text, text to speech, and natural language processing to ensure effective communication between patients and the virtual assistant avatar. Overall, the system and method is designed to improve remote patient monitoring, provide more efficient and effective communication between patients and healthcare professionals, and facilitate better overall patient outcomes.
###"""

# API logic
@app.route('/api/get_data', methods=['POST'])
def get_data():
    data = request.get_json() 
    keywords = [keyword.strip() for keyword in data["input_text"].split(",")]
    
    #Creating the prompt
    def create_prompt(training_cases, keywords):
        keywords = ", ".join(keywords)
        prompt = training_cases + "\nKeywords: "+ keywords+ "\nOutput:"
        return prompt
    prompt = create_prompt(training_cases, keywords)

    # Generate text using the HappyGeneration model
    args_beam = GENSettings(num_beams=5, no_repeat_ngram_size=2, early_stopping=True, min_length=5, max_length=150)
    # Generate text using the HappyGeneration model
    output_text= happy_gen.generate_text(prompt, args=args_beam)

    # Return the generated text as a JSON response
    print("output_text:", output_text)
    return jsonify({"output_text": output_text.text})


if __name__ == '__main__':
  app.run()