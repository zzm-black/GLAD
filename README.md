
<h1 align="center">Generative LLM-agents for Application Development (GLAD)</h1>

[**Live Demo**](https://solitairevue.firebaseapp.com)

## Contributions are Welcome!
Interested in contributing? Check out the [Contribution Guidelines](https://github.com/silent-lad/VueSolitaire/blob/master/CONTRIBUTING.md).

## Project Overview

GLAD is a generative LLM-agent system designed to streamline task management and automate the workflow for Kanboard API-based applications. Built with LangChain and integrated with OpenAI’s models, GLAD uses structured prompts and API methods to manage complex, multi-step processes within Kanboard.

![Demo GIF of GLAD in Action](https://media.giphy.com/media/7OWdOQupgCClrZb19P/giphy.gif)

### Project Features
- **Automated Task Splitting:** Dynamically splits a complex task into actionable subtasks.
- **API Parameter Identification:** Extracts parameters required for executing subtasks via the Kanboard API.
- **Method Selection for Execution:** Matches subtasks with the most suitable API methods based on their parameters.
- **Streamlined Execution with Interactive UI:** GLAD is wrapped in a Streamlit interface, enabling users to submit task descriptions, view subtasks, and check results.

## Project Details

### Task Splitting and API Interaction
- **TaskSplitter:** A custom LangChain tool that decomposes a complex task into logical, ordered subtasks. Each step is assessed for API compatibility.
- **Parameter Identification:** Identifies parameters for executable subtasks, structured by task type (project, column, or task).
- **Method Selector:** Automatically selects the best-suited Kanboard API method for each identified task type and parameter set.
  
### Drag-and-Drop Interface
Implemented with Streamlit, the interface supports the following:
- **Task Submission**: Users input task descriptions to generate Kanboard-compatible workflows.
- **Interactive Debugging**: View and validate task decomposition, parameter extraction, and method selection in real-time.

### CSS Styling and UI
- Custom CSS styling is applied for an intuitive, card-like display of tasks and results.
  
## Getting Started

### Installation

To set up the project locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Environment Variables
Add your OpenAI and Kanboard API credentials to an `.env` file:

```plaintext
KANBOARD_API_URL="http://localhost:8081/jsonrpc.php"
API_TOKEN="YOUR_API_TOKEN_HERE"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

## Usage

### Drag-and-Drop Steps
1. **Input Task Description**: Enter a high-level task in the Streamlit interface.
2. **Task Splitting**: GLAD decomposes the task into subtasks, identifies parameters, and matches each with the appropriate API method.
3. **Execution and Results**: View the API calls made, responses received, and any necessary follow-up actions.

### Example Task Workflow
1. Enter a complex task in the input box.
2. Observe the process breakdown in four steps: Task Splitting, Parameter Identification, Method Selection, and API Execution.

### Supported API Methods
GLAD currently supports the following Kanboard API methods:

- `createProject`
- `getProjectByName`
- `getBoard`
- `createTask`
- `closeTask`
- `moveTaskPosition`

## Future Improvements
- **Enhanced Error Handling**: Improve responses for API errors or task failures.
- **New Task Templates**: Add templates for commonly used task workflows.
- **Drag-and-Drop for Tasks**: Enhance the UI with real-time drag-and-drop capabilities for task management.

## Support the Project
If you find this project helpful, consider supporting us with a donation!
