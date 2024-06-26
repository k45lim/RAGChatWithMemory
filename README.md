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
