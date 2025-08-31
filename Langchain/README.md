# [Evolution of AI (Artificial Intelligence)]()

1. **Origin**

   * Concept of AI started in the 1950s.
   * 1956: Term "Artificial Intelligence" coined at the Dartmouth Conference.

2. **Early AI (1950s–1970s)**

   * Focus on symbolic AI and rule-based systems.
   * Programs designed for games, logical reasoning, and problem-solving.
   * Expert systems developed for narrow decision-making tasks.

3. **AI Winter (1980s)**

   * Progress slowed due to high expectations and limited computing power.
   * Funding and interest dropped.

4. **Rise of Machine Learning (1990s)**

   * Shift from rule-based to data-driven approaches.
   * Systems began learning from data rather than hard-coded rules.
   * 1997: IBM Deep Blue defeated chess world champion Garry Kasparov.

5. **Big Data and Deep Learning (2000s–2010s)**

   * More data and better hardware enabled progress.
   * 2012: Neural networks won ImageNet competition — breakthrough for deep learning.
   * AI used in vision, speech, and recommendation systems.

6. **Milestone in Reinforcement Learning**

   * 2016: DeepMind’s AlphaGo beat the world champion in Go using deep reinforcement learning.

7. **Advances in Language Understanding**

   * Models like Word2Vec, GloVe, and BERT introduced contextual understanding of language.
   * Enabled more accurate NLP tasks like translation, summarization, and Q\&A.

8. **Transformer Era (2017 Onwards)**

   * Transformers introduced in paper "Attention is All You Need" (Google, 2017).
   * Allowed parallel processing and better understanding of sequence data.

9. **Rise of LLMs (Large Language Models)**

   * 2018: OpenAI launched GPT-1.
   * 2019: GPT-2 showed powerful text generation but was partially held back due to misuse concerns.
   * 2020: GPT-3 launched with 175B parameters; showed strong generalization.
   * 2022: ChatGPT launched, making LLMs accessible to public.
   * 2023: GPT-4 released with multimodal abilities.

10. **Open Source and Ecosystem Growth**

* Meta (LLaMA), Mistral, and other open-source LLMs emerged.
* Tools like LangChain developed to make LLMs useful in real-world apps (e.g., chatbots, agents, automation).

11. **Present**

* AI is widely used in customer support, education, content creation, programming, healthcare, and research.

### [Why LangChain was built?]()

When people started using powerful language models like GPT-3, they noticed something — the models were smart, but they didn’t follow structured plans, couldn’t remember past chats, and had no idea how to use external tools like search engines, APIs, or databases.

So engineers kept hitting a wall:
*“The model is brilliant… but how do I make it actually *do* something useful in a real app?”*

That’s where **LangChain** came in.

It was built to solve a simple but deep problem:

> How can we connect LLMs to **real-world data, memory, and actions** — and do it in a **repeatable, flexible, developer-friendly** way?

The goal was never just to chat with the model. It was to:

* **Give it memory** (so it can remember context across turns).
* **Let it use tools** (like a calculator, search, or your company’s data).
* **Design workflows** (chains of steps it can follow).
* **Build agents** (that decide what to do next, not just predict the next word).
* **Make it production-ready** (with logging, evaluation, deployment).

LangChain wasn’t just a coding library — it was an answer to the question:
**"How do we move from playing with LLMs to building serious, useful applications with them?"**


# [LangChain Ecosystem]()
### 1. **LangChain Core Concepts**

**→ LLMs and Chat Models**

* Interfaces for calling language models (e.g., OpenAI, Anthropic, Cohere, Hugging Face).
* Supports both completion-style models (`LLM`) and structured chat interfaces (`ChatModel`).

**→ Prompts**

* Templates used to format input for LLMs.
* Two types: `PromptTemplate` (static text + variables) and `ChatPromptTemplate` (for multi-turn conversations).
* Enables reusability and modular design in prompt engineering.

**→ Chains**

* Logic that links components together (e.g., prompt → model → output).
* `SimpleChain`, `LLMChain`, and `SequentialChain` are common.
* Helps create workflows that involve multiple steps or calls.

**→ Memory**

* Stores past interactions to enable conversational context.
* Types: `BufferMemory`, `SummaryMemory`, `ConversationTokenBufferMemory`, etc.
* Crucial for chatbots and agent continuity.

**→ Tools**

* External functions LLMs can call via agents.
* Examples: search, calculator, database queries, APIs.
* Tools are registered and invoked when the LLM is unsure or needs real-world data.

**→ Agents**

* LLMs that can decide which tools to use to solve a task.
* Use ReAct, Plan-and-Execute, or other decision frameworks.
* Example: `initialize_agent()` sets up an agent with tools and LLM.

**→ Callbacks**

* Used for logging, tracing, debugging.
* Helps track the flow of execution across complex chains or agents.

**→ Output Parsers**

* Help convert LLM output into structured formats like JSON, lists, or Python objects.
* Useful in programmatic applications that require reliable formatting.

**→ Runnables (LangChain Expression Language – LCEL)**

* New abstraction that makes pipelines declarative and composable.
* Supports `.invoke()`, `.batch()`, `.stream()`.
* Core to the new version of LangChain (LangChain v0.1+).



### 2. **Retrieval-Augmented Generation (RAG)**

**→ Documents**

* Raw data from your own source (PDFs, websites, docs, etc.).

**→ Loaders**

* Used to load documents from files, URLs, APIs.
* Example: `PyPDFLoader`, `WebBaseLoader`, `CSVLoader`.

**→ Text Splitters**

* Split documents into chunks for better retrieval and embedding.
* Chunk size and overlap are configurable.

**→ Embeddings**

* Vector representations of text using models like OpenAI, HuggingFace, Cohere.
* Used to match questions with relevant chunks.

**→ Vector Stores**

* Databases that store embeddings and allow similarity search.
* Examples: FAISS, Pinecone, Chroma, Weaviate, Qdrant.

**→ Retrievers**

* Fetch relevant documents based on input query.
* Can be combined with filters, metadata, or hybrid search.

**→ RAG Chains**

* Use retrievers to get context, then pass that into an LLM with a prompt.
* Supports better factual accuracy and domain-specific answers.


### 3. **LangServe**

* A fast way to turn LangChain pipelines into REST APIs.
* Works with FastAPI.
* Ideal for deploying chains, agents, or RAG pipelines.



### 4. **LangSmith**

* A developer platform for debugging, testing, evaluating, and monitoring LangChain apps.
* Logs every step, prompt, output, and LLM call.
* Supports human and automated evaluation of outputs.

### 5. **LangGraph (Advanced)**

* A stateful multi-actor framework based on event-driven graphs.
* Used to create complex workflows, memory-aware agents, and multi-turn systems.
* Especially useful for long-running or looped agent tasks.



### 6. **Supported Integrations**

* **LLMs**: OpenAI, Anthropic, Cohere, HuggingFace, Mistral, Together, Groq, etc.
* **Vector DBs**: Pinecone, FAISS, Chroma, Weaviate, Qdrant, Milvus.
* **Tools**: SerpAPI, Tavily, DuckDuckGo, WolframAlpha, Python REPL, Zapier, SQL, etc.


### 7. **Common Use Cases**

* Chatbots with memory and custom knowledge.
* RAG pipelines for answering from private documents.
* Agents that use tools to solve user goals.
* Automation tools that interact with APIs and databases.
* Evaluators for testing prompt/model performance.

# [LangChain – Models]()
<h3>1. <strong>Open Source vs Closed Source Models</strong></h3>
<h4><strong>Closed Source Models</strong></h4>
<ul>
<li>
<p>Proprietary models hosted by companies.</p>
</li>
<li>
<p>Accessed via API — you don't get the model weights.</p>
</li>
<li>
<p>High performance, optimized, and often state-of-the-art.</p>
</li>
<li>
<p>Great for production use due to reliability and scalability.</p>
</li>
<li>
<p><strong>Examples</strong>:</p>
<ul>
<li>
<p>GPT-3.5, GPT-4, GPT-4o (OpenAI)</p>
</li>
<li>
<p>Claude (Anthropic)</p>
</li>
<li>
<p>Gemini (Google)</p>
</li>
<li>
<p>Bedrock models (Amazon)</p>
</li>
<li>
<p>Cohere's Command R+ (API-only version)</p>
</li>
</ul>
</li>
</ul>
<h4><strong>Open Source Models</strong></h4>
<ul>
<li>
<p>Model weights are publicly released — can be run locally or fine-tuned.</p>
</li>
<li>
<p>Offers full control, privacy, and customization.</p>
</li>
<li>
<p>Often used in research, offline apps, or cost-sensitive scenarios.</p>
</li>
<li>
<p>Hosted on platforms like HuggingFace, Together, Fireworks, etc.</p>
</li>
<li>
<p><strong>Examples</strong>:</p>
<ul>
<li>
<p>LLaMA 2, LLaMA 3 (Meta)</p>
</li>
<li>
<p>Mistral, Mixtral</p>
</li>
<li>
<p>Falcon (TII)</p>
</li>
<li>
<p>Gemma (Google), Phi-3 (Microsoft)</p>
</li>
<li>
<p>OpenChat, Zephyr, Command R (open-weight)</p>
</li>
</ul>
</li>
</ul>
<hr>
<h3>2. <strong>Model Types in LangChain</strong></h3>
<p>LangChain provides unified interfaces to work with different model types:</p>
<h4>→ <strong>LLM (Language Model)</strong></h4>
<ul>
<li>
<p>Used for single-turn text generation (prompt → response).</p>
</li>
<li>
<p>Example: GPT-3, Mistral, LLaMA, Cohere.</p>
</li>
</ul>
<h4>→ <strong>Chat Model</strong></h4>
<ul>
<li>
<p>Used for role-based, multi-turn conversations.</p>
</li>
<li>
<p>Input as list of messages with roles: <code inline="">system</code>, <code inline="">user</code>, <code inline="">assistant</code>.</p>
</li>
<li>
<p>Example: ChatGPT, Claude, LLaMA Chat, Zephyr.</p>
</li>
</ul>
<h4>→ <strong>Embedding Model</strong></h4>
<ul>
<li>
<p>Converts text into high-dimensional vectors for similarity search.</p>
</li>
<li>
<p>Used heavily in Retrieval-Augmented Generation (RAG).</p>
</li>
<li>
<p>Example: OpenAI Embeddings, HuggingFace Transformers, Cohere Embed.</p>
</li>
</ul>
<h4>→ <strong>Multi-modal Models</strong> (Advanced)</h4>
<ul>
<li>
<p>Accept and generate across multiple data types: text, images, etc.</p>
</li>
<li>
<p>Example: GPT-4 Vision, Gemini 1.5 Pro.</p>
</li>
</ul>
<hr>
<h3>3. <strong>How LangChain Works with Models</strong></h3>
<p>LangChain abstracts all models under a standard interface, so developers can switch between providers easily.</p>
<h4> Benefits:</h4>
<ul>
<li>
<p>Plug-and-play model swapping</p>
</li>
<li>
<p>Compatible with tools, agents, memory</p>
</li>
<li>
<p>Enables standardized logging, streaming, and error handling</p>
</li>
</ul>
<h4> Supported Features:</h4>
<ul>
<li>
<p><code inline="">.predict()</code> or <code inline="">.invoke()</code> for synchronous outputs</p>
</li>
<li>
<p><code inline="">.stream()</code> for streaming tokens</p>
</li>
<li>
<p><code inline="">.batch()</code> for parallel calls</p>
</li>
<li>
<p>Callbacks for tracing and debugging</p>
</li>
</ul>
<hr>
<h3>4. <strong>Popular Providers in LangChain</strong></h3>

Provider | Type | Open/Closed | Notes
-- | -- | -- | --
OpenAI | Chat, LLM, Embedding | Closed | GPT-4, GPT-4o, Embeddings
Anthropic | Chat | Closed | Claude 2, 3
Google (VertexAI) | Chat, Embedding | Closed | Gemini
Cohere | LLM, Embedding | Mixed | Command R (open & closed)
Meta | LLM, Chat | Open | LLaMA 2, LLaMA 3
Mistral | LLM, Chat | Open | Mistral-7B, Mixtral
Hugging Face | LLM, Embed | Open | Thousands of open models
Together AI | Any | Open-hosted | Runs open models via API
Fireworks AI | Any | Open-hosted | Focus on hosted open weights
Groq | Chat, LLM | Open-hosted | Extremely fast inference


<hr>
<h3>5. <strong>Best Practices for GenAI Engineers</strong></h3>
<ul>
<li>
<p>Use <strong>closed-source models</strong> when you need performance, scale, and safety.</p>
</li>
<li>
<p>Use <strong>open-source models</strong> for control, privacy, and custom deployments.</p>
</li>
<li>
<p>Combine with <strong>embedding models</strong> for search and retrieval (RAG).</p>
</li>
<li>
<p>Use LangChain’s standard model wrappers to future-proof your architecture.</p>
</li>
<li>
<p>Always monitor model usage (tokens, latency, costs) using LangChain callbacks or LangSmith.</p>
</li>
</ul>


# [Prompts in LangChain]()

A **prompt** is an instruction or query given to a language model to guide its output. It defines what the model should do — like answering a question, summarizing text, writing code, etc.

* In LangChain, prompts are structured to ensure consistency, flexibility, and control over model behavior.
* Effective prompting is the foundation of **prompt engineering** — a key GenAI skill.


## **Types of Prompts**

### 1. **Static Prompt**

* A fixed, pre-written instruction.
* Same every time you run the code — no dynamic input.
* Example: `"Explain the history of AI."`

### 2. **Dynamic Prompt**

* Created at runtime using variables.
* Allows inserting user input, database values, or context.
* Enables dynamic applications like custom Q\&A, search, chatbots, etc.


## **What is PromptTemplate?**

A `PromptTemplate` is a tool provided by LangChain to help you construct dynamic prompts for text-based LLMs (like GPT-3 or LLaMA).

**Why do we use it?**

* To insert variables like `{name}` or `{topic}` into a prompt safely and cleanly
* To make prompts reusable across different use cases
* To enable validation (ensures required fields are filled)

```python
from langchain.prompts import PromptTemplate

template = "Translate the following to French:\n{sentence}"
prompt = PromptTemplate(input_variables=["sentence"], template=template)
```



## **What is ChatPromptTemplate?**

`ChatPromptTemplate` is used when working with **chat models** (like GPT-4, Claude, Gemini) that accept **structured role-based messages** instead of a single string.

**Why do we use it?**

* To define multiple messages with roles: `system`, `user`, and `assistant`
* To build multi-turn conversations with memory and history
* To inject chat context or session history using placeholders

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])
```



## **Static Messages in ChatPromptTemplate**

Used when you define the full prompt in advance:

1. **System Message**: sets assistant behavior
2. **Human Message**: user input
3. **AI Message**: optional, previous model reply


## **Dynamic Messages with MessagesPlaceholder**

Sometimes you want to inject **chat history** or previous messages dynamically.

Use `MessagesPlaceholder` for that:

```python
from langchain.prompts import MessagesPlaceholder

ChatPromptTemplate.from_messages([
    ("system", "You are a support bot."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{current_question}")
])
```

**Why use it?**
It helps carry context across conversations — useful in real apps like refund tracking, customer support, or tutoring bots.

---

## **Invoke Flow in LangChain**

* Static prompt → direct string
* Dynamic prompt → `PromptTemplate.format()`
* Multi-turn prompt → `ChatPromptTemplate` with role-based messages
* Use `.invoke()` to send the formatted result to the model

---

## **Important Points to Remember**

* **PromptTemplate** is used with plain LLMs (text-only input/output).
* **ChatPromptTemplate** is used with ChatModels (message-based input).
* Use **PromptTemplate** for single-turn tasks like summarization, generation, or rewriting.
* Use **ChatPromptTemplate** for bots, multi-turn agents, and memory-based apps.
* Use **MessagesPlaceholder** to insert memory or chat history dynamically.
* Always design prompts clearly — vague prompts confuse the model and reduce performance.

# Structured Outputs in LangChain 

## Overview
Structured output generation enables language models to return data in predefined formats (JSON, XML, etc.) rather than unstructured text. This approach enhances interoperability with downstream systems and applications.

## Core Concepts

### 1. Structured vs Unstructured Outputs
| Feature | Structured Output | Unstructured Output |
|---------|------------------|-------------------|
| Format | Machine-readable (JSON/XML) | Free-form text |
| Parsing | Directly consumable | Requires NLP processing |
| Validation | Schema-enforced | No validation |
| Use Case | System integration | Human consumption |

### 2. Implementation Components
- **Output Parsers**: Transform LLM responses into structured formats
- **Pydantic Models**: Define and validate output schemas
- **Prompt Engineering**: Special instructions for format compliance

## Technical Implementation

### Example: Resume Data Extraction
```python
from pydantic import BaseModel
from typing import List

class CandidateProfile(BaseModel):
    name: str
    contact: str
    skills: List[str]
    experience: float  # in years

# LLM prompt includes:
# "Return response as JSON matching this schema: {schema}"
```

### Output Transformation Process
1. Define schema using Pydantic
2. Configure output parser
3. Structure prompt instructions
4. Parse and validate LLM response

## Key Use Cases

### 1. Automated Data Processing
- Extract structured information from documents
- Standardize data collection pipelines
- Enable database population directly from LLM outputs

### 2. API Development
- Generate consistent API responses
- Validate request/response payloads
- Automate documentation generation

### 3. Agent Systems
- Standardize tool outputs
- Enable predictable inter-agent communication
- Facilitate error handling in workflows

## Best Practices

1. **Schema Design**
   - Maintain backward compatibility
   - Include field descriptions
   - Define optional vs required fields

2. **Validation**
   - Implement strict type checking
   - Set reasonable value constraints
   - Provide clear error messages

3. **Performance**
   - Balance schema complexity with usability
   - Cache parsed schemas
   - Monitor parsing failures

## Advanced Features

### Dynamic Schema Generation
```python
def create_schema(fields: Dict[str, type]):
    return create_model('DynamicSchema', **fields)
```

### Schema Versioning
```python
class ProfileV2(CandidateProfile):
    certifications: List[str] = []
```

## Integration Patterns

1. **Database Integration**
   - Direct ORM mapping
   - Bulk insert operations
   - Change data capture

2. **API Gateways**
   - Request/response validation
   - Schema-based routing
   - Response transformation

3. **Stream Processing**
   - Schema registry integration
   - Serialization/deserialization
   - Schema evolution handling

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Validation errors | Check field types and constraints |
| Missing fields | Verify prompt instructions |
| Format mismatches | Test with simpler schemas first |
| Performance issues | Optimize schema complexity |

## Conclusion
Structured outputs transform LLMs from text generators to predictable data processors. When implemented properly, they:
- Reduce integration complexity
- Improve system reliability
- Enable new automation scenarios
- Lower maintenance costs

Adopting this approach is particularly valuable for production systems requiring consistent, machine-readable outputs from language models.


#  [Output Parsers in LangChain]()

## **What is an Output Parser?**

An **Output Parser** in LangChain is used to **process the raw output** generated by a language model and convert it into a **structured, usable format**.

By default, LLMs return plain text — but most real-world applications need output in formats like:

* JSON
* List
* Dictionary
* Custom objects (e.g., class instances)



## **Why Do We Use Output Parsers?**

* **Convert free-form text into structured data** (for automation or downstream logic)
* **Validate** the format of the output (e.g., is it valid JSON?)
* **Ensure consistency** in responses (especially when LLMs hallucinate or deviate)
* **Integrate easily** with chains, agents, or external systems



## **Common Types of Output Parsers in LangChain**

### 1. **StrOutputParser**

* Returns plain string output from the LLM
* Used when the output is just natural language

```python
from langchain.output_parsers import StrOutputParser

parser = StrOutputParser()
```



### 2. **CommaSeparatedListOutputParser**

* Converts text like `"apple, banana, cherry"` → `["apple", "banana", "cherry"]`
* Useful for classification, tags, lists

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
```



### 3. **PydanticOutputParser**

* Converts output to a **Pydantic model** (data class)
* Ensures strict schema and validation
* Used for typed applications, APIs, or structured form responses

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Info(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Info)
```



### 4. **JsonOutputParser**

* Expects output in proper JSON format
* Best when using JSON prompt templates (e.g., tool-using agents)
* Often combined with `format_instructions` in prompt



## **Best Practice: Combine with Format Instructions**

Always include **clear format instructions** in your prompt so the LLM knows exactly what structure you expect.

```python
prompt = PromptTemplate(
    template="Give me a list of fruits in comma-separated format.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```



## **When Are Output Parsers Most Useful?**

* Building **tool-using agents**
* Creating **form-fillers** or **structured reports**
* Developing **QA systems** with structured outputs
* Validating input/output **before database entry**



## **Important Points to Remember**

* Output parsers help **bridge the gap** between LLM’s creative text and structured application logic
* Use **PydanticOutputParser** for strong typing and schema enforcement
* Use **JsonOutputParser** when your downstream system expects machine-readable output
* Combine parsers with **format instructions** for best results
* Always **test and validate** outputs — LLMs are powerful, but may still break formatting rules

# [Chains in LangChain]()

## **What is a Chain?**

A **Chain** in LangChain is a **pipeline that connects multiple components** (like prompts, models, output parsers, tools) to build a complete GenAI application.

It connects all steps from **user input** to **final structured output** — enabling custom logic, multi-step reasoning, and tool usage.


## **LLM → App Pipeline: How It Works**

```
User → Prompt → LLM → Response → Processing → Output
```

LangChain helps stitch this into a clean, reusable **chain**, like this:

```python
Input → PromptTemplate → LLM → OutputParser → Final Result
```

Each block is modular and can be replaced or extended.


## **Why Use Chains?**

* Combine multiple steps into a single flow
* Build apps that need pre- and post-processing
* Orchestrate multiple LLM calls
* Handle memory, tools, documents, APIs, etc.
* Useful for both simple and complex workflows

## **Types of Chains**

### **1. Parallel Chains**

**What:** Multiple LLMs run in parallel on the same or different inputs.

**Use Case:** Suppose you upload a PDF. You want:

* Notes generated from the PDF
* A quiz created from the notes
* Both outputs merged into one response

**Implementation:**

* Chain 1 → OpenAI model → Generates Notes
* Chain 2 → Claude model → Generates Quiz
* Chain 3 → Merges both outputs into a final summary using a 3rd model

→ All chains run **in parallel** and results are then combined.

### **When to Use:**

* Multi-output generation
* Different tasks needing different models
* Saving time by parallel processing

### **2. Sequential Chains**

**What:** Chains run **step by step**, passing output of one to the next.

**Use Case:** Build a writing assistant:
→ Step 1: Generate blog topic
→ Step 2: Generate outline from topic
→ Step 3: Write full blog using outline

Each LLM task feeds the next — just like function composition.

**Implementation:**
Chain 1 → Chain 2 → Chain 3

### **When to Use:**

* Multi-step reasoning
* Tasks where one result depends on previous output
* Logical workflows (e.g., summarize → translate → analyze)


### **3. Conditional Chains**

**What:** Logic-based chain that chooses the next step based on conditions or input type.

**Use Case:**
→ If user input is a **question**, use QA chain
→ If input is a **document**, use summarization chain
→ If it’s a **command**, use agent/tool chain

**Implementation:**

```python
if "?" in input:
    run_QA_chain()
else:
    run_summary_chain()
```

### **When to Use:**

* Smart assistants
* Chatbots with decision branches
* Agentic workflows that adapt dynamically


## **Important Points to Remember**

* **Chain = LLM + Logic**
* Use **Sequential Chains** for step-by-step flows
* Use **Parallel Chains** for generating multiple outputs at once
* Use **Conditional Chains** when logic decides the flow
* Chains are building blocks for **Agents**, **Apps**, **Tools**
* Always include **prompt templates** and **output parsers** inside chains for clean flow

# [Runnables in LangChain]()
<h2><strong>1. Why Runnables Were Introduced ?</strong></h2>
<ul>
<li>
<p>After OpenAI released LLM APIs in 2022, GenAI app development grew rapidly.</p>
</li>
<li>
<p>Developers had to manually connect many steps: loading, splitting, embedding, prompting, parsing, etc.</p>
</li>
<li>
<p>Each tool or model had different interfaces — no standard structure.</p>
</li>
<li>
<p>LangChain tried to help by offering components — but they had inconsistent APIs.</p>
</li>
<li>
<p>This made learning LangChain <strong>harder</strong>, especially for beginners.</p>
</li>
<li>
<p>Developers had to write <strong>custom chains</strong> just to connect basic components.</p>
</li>
<li>
<p>The solution: <strong>Runnables</strong> — a unified, pluggable interface for all steps.</p>
</li>
</ul>

<h2><strong>2. What Is a Runnable</strong></h2>
<ul>
<li>
<p>A <strong>Runnable</strong> is a <strong>standard unit of work</strong> in LangChain.</p>
</li>
<li>
<p>It follows a common structure:<br>
<code inline="">Input → Process → Output</code></p>
</li>
<li>
<p>Every step in your GenAI pipeline is treated as a Runnable:<br>
loaders, splitters, LLMs, prompts, retrievers, agents.</p>
</li>
</ul>
<hr>
<h2><strong>3. Why Runnables Are Useful</strong></h2>
<ul>
<li>
<p>They give all components the <strong>same interface</strong> — solving the compatibility issue.</p>
</li>
<li>
<p>Runnables are <strong>chainable</strong> — output of one becomes input for the next.</p>
</li>
<li>
<p>Every Runnable has:</p>
<ul>
<li>
<p><code inline="">.invoke()</code> → For single input</p>
</li>
<li>
<p><code inline="">.batch()</code> → For list of inputs</p>
</li>
<li>
<p><code inline="">.stream()</code> → For streamed output</p>
</li>
<li>
<p><code inline="">.transform()</code> → For iterable inputs</p>
</li>
</ul>
</li>
</ul>
<h2><strong>4. How Chains Work With Runnables</strong></h2>
<ul>
<li>
<p>A <strong>Chain</strong> is just a <strong>sequence of Runnables</strong> connected together.</p>
</li>
<li>
<p>Example chain:</p>
<pre><code>Document → [Loader] → [TextSplitter] → [Embedder] → [Retriever] → [PromptTemplate] → [LLM] → Output
</code></pre>
</li>
<li>
<p>Each block is a Runnable, and the full workflow is also a Runnable.</p>
</li>
<li>
<p>You can reuse or replace any part — just like Lego blocks.</p>
</li>
</ul>

<h2><strong>5. Examples of Common Runnables</strong></h2>

Component | Role
-- | --
Document Loader | Loads raw text or files
Text Splitter | Breaks text into chunks
Embedder | Converts text into vectors
Vector Store | Stores/retrieves embeddings
PromptTemplate | Formats input for LLM
ChatModel / LLM | Generates response
OutputParser | Parses raw model output


<hr>
<h2><strong>6. Real Benefit</strong></h2>
<ul>
<li>
<p>LangChain made a mistake initially:<br>
→ They built components without a <strong>shared interface</strong>.</p>
</li>
<li>
<p>This led to custom wrappers, messy chains, and frustration.</p>
</li>
<li>
<p><strong>Runnables fixed this</strong> by giving:</p>
<ul>
<li>
<p>A <strong>clean, plug-and-play system</strong></p>
</li>
<li>
<p><strong>Standardized API</strong> across all blocks</p>
</li>
<li>
<p>Easier debugging and unit testing</p>
</li>
<li>
<p>Seamless integration between components</p>
</li>
</ul>
</li>
</ul>
<hr>
<h2><strong>7. Real-World Analogy</strong></h2>
<p>Think of Runnables like <strong>Lego blocks</strong>:</p>
<ul>
<li>
<p>All have the same connectors.</p>
</li>
<li>
<p>You can build anything by snapping them together.</p>
</li>
<li>
<p>Replace one piece without breaking the rest.</p>
</li>
</ul>
<hr>
<h2><strong>Key Points to Remember</strong></h2>
<ul>
<li>
<p>Runnable = unit of work with input/output logic</p>
</li>
<li>
<p>Chains = sequence of Runnables</p>
</li>
<li>
<p>All Runnables follow the same methods: <code inline="">.invoke()</code>, <code inline="">.batch()</code>, <code inline="">.stream()</code></p>
</li>
<li>
<p>You can compose, nest, and reuse workflows easily</p>
</li>
<li>
<p>Modern LangChain is built entirely around Runnables</p>
</li>
</ul>


# [Tools in LangChain]()

### **What is a Tool?**

* A **Tool** in LangChain is a wrapper around any function or API that the LLM **can call** to complete a task.
* Tools extend the capability of LLMs — from just answering questions to **acting** (e.g., search the web, do math, fetch data, call APIs).

Think of tools as giving “hands” to the LLM.


### **What Can Tools Do?**

* Fetch real-time weather
* Query a database
* Search documents
* Use a calculator
* Hit a REST API
* Trigger a custom Python function
* Access code execution or file systems



## **How to Create a Tool in LangChain (3 Approaches)**



### **1. Using `@tool` decorator** (Quick and Simple)

Best for: Wrapping small Python functions as tools.

```python
from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    return a * b
```

* Automatically creates a tool with name, description, input/output schema.
* Easy to plug into agents.
* Good for quick use-cases.



### **2. Using the `Tool` class manually** (Custom Control)

Best for: When you want full control over the tool config.

```python
from langchain.tools import Tool

def greet(name: str) -> str:
    return f"Hello, {name}!"

custom_tool = Tool(
    name="greet_user",
    func=greet,
    description="Greets a user with their name."
)
```

* Lets you define name, function, and detailed description.
* Better if you're building tools dynamically or customizing behavior.



### **3. Using `StructuredTool`** (For functions with multiple inputs)

Best for: Tools that take **structured inputs** (like multiple arguments with types).

```python
from langchain.tools import StructuredTool

def convert_currency(amount: float, currency: str) -> str:
    return f"{amount} USD converted to {currency}"

tool = StructuredTool.from_function(convert_currency)
```

* Supports input validation using function signatures.
* Enables more complex tool logic with multiple parameters.



### **Tool Usage in LangChain Agents**

* Tools are passed into **agents**, which use them **dynamically** based on the user's query.
* LangChain agents decide *when* and *which* tool to use depending on the prompt.

Example:

> "What's the weather in Delhi and then multiply that temperature by 2"
> → Agent calls weather tool first, then math tool.

## **Important Points **

* Tools allow **LLMs to take actions**, not just generate text.
* `@tool` is fastest to use; `StructuredTool` is best for multiple input fields.
* Tools are **essential for building agentic systems**.
* Always include a good `description` so the LLM knows **when** to use the tool.
