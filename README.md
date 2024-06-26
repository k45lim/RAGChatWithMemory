# RAGChatWithMemory
End-to-end RAG application with various document types and conversation with chat memory

Challenge/requirements:
1. Take at least 5 data sources (1 powerpoint, 1 pdf, 1 textfile, 1 csv and 1 web site)
2. Maintain one knowledge base (choose one from mongodb, astradb, pinecone or weviate)
3. Using prompt, user enters a question and the application should answer using what is stored in the knowledge database
4. Handle common question like, hi, hello, good morning, good evening etc.
5. You have to mention the complete memory of the conversation (threshold is 10)
6. Create a UI for the chat bot

Proposed solution:
1. Create a application (jupyter notebook) that will read from a source directory that contains the require document types, and chunk them, embed and store the embedded data to Pinecone vector database.  The source directory is at './source' and contains 1 file per type except for websites which read and load the url/s contained in the 'urllist.txt'
2. Once the data has been successfully propagated to the vector store in the cloud (Pinecode), run a streamlit application that initially connect to the vector store, allow the user to set the number of messages to track for chat history and prompt the user to enter his/her questions.  The application will display only the number of messages set using the slider and memory of the conversation is also limited according to the set limit.

Applications:
1. Dataset Creation.ipynb - created in jupyter notebook for data ingestion, reading all files in the directory './source'.  It connects to Pinecone (using its latest implementation of serverless aws platform).  It uses local llm Ollama/Llama3 both for execution and data embedding.

2. appRagConversation.py - should be run using 'streamlit run appRagConversation.py'.  Offers the ability to limit the conversation memory from 1 to 20 which is visible as well in the main display.  It uses local llm Ollama/Llama3 both for execution and data embedding.

Data used: 
1. All files used are inside the sub-directory 'source'.
2. Here are the brief description of the files:
	a. country_code.csv - an comma delimited file containing country code on telecommunication per country (eg. +1 for Canada)
	b. Juan Dela Cruz Resume.pdf - a resume of fictitious person
	c. Project Management Framework and Tools MASTER COPY.pptx - a powerpoint presentation material about project management
	d. State Of The Nation.txt - a textfile containing the state of the nation address of the USA president
	e. URLList.txt - a text the contains the URLs to be read, loaded and embedded to vector store. The contents are:
		- https://en.wikipedia.org/wiki/Magnus_Carlsen - details about the current world's number 1 rated Chess player. 
		- https://www.radioddity.com/blogs/all/shortwave-radio - details about shortwave band radio