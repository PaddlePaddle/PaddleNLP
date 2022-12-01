import { CodeEditor } from '@jupyterlab/codeeditor';
import { DataConnector } from '@jupyterlab/statedb';
import { CompletionHandler } from '@jupyterlab/completer';
import { requestAPI } from './handler';

/**
 * Get code from backend.
 * @param prompt
 */
async function codeGen(prompt: string): Promise<string> {
  const dataToSend = { prompt: prompt };
  try {
    const postCode = await requestAPI<any>('codegen', {
      body: JSON.stringify(dataToSend),
      method: 'POST'
    });
    console.log(postCode);
    return postCode.res;
  } catch (reason) {
    console.error(
      `Error on POST /codegen-paddle-backend/codegen ${dataToSend}.\n${reason}`
    );
    return '';
  }
}

/**
 * A custom connector for completion handlers.
 */
export class CustomConnector extends DataConnector<
  CompletionHandler.IReply,
  void,
  CompletionHandler.IRequest
> {
  /**
   * Create a new custom connector for completion requests.
   *
   * @param options - The instatiation options for the custom connector.
   */
  constructor(options: CustomConnector.IOptions) {
    super();
    this._editor = options.editor;
  }

  /**
   * Fetch completion requests.
   *
   * @param request - The completion request text and details.
   * @returns Completion reply
   */
  fetch(
    request: CompletionHandler.IRequest
  ): Promise<CompletionHandler.IReply> {
    if (!this._editor) {
      return Promise.reject('No editor');
    }
    return new Promise<CompletionHandler.IReply>(resolve => {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      resolve(Private.completionHint(this._editor!));
    });
  }

  private readonly _editor: CodeEditor.IEditor | null;
}

/**
 * A namespace for custom connector statics.
 */
export namespace CustomConnector {
  /**
   * The instantiation options for cell completion handlers.
   */
  export interface IOptions {
    /**
     * The session used by the custom connector.
     */
    editor: CodeEditor.IEditor | null;
  }
}

/**
 * A namespace for Private functionality.
 */
namespace Private {
  /**
   * Get a list of mocked completion hints.
   *
   * @param editor Editor
   * @returns Completion reply
   */
  export async function completionHint(
    editor: CodeEditor.IEditor
  ): Promise<CompletionHandler.IReply> {
    // Find the token at the cursor
    const cursor = editor.getCursorPosition();
    const cursorToken = editor.getTokenForPosition(cursor);

    let haveComment = false;
    let lastLine = '';
    const thisLine: string = editor.getLine(cursor.line) as string;
    let autoGenCode: string;

    // Confirm whether there are comments
    if (cursor.line !== 0) {
      lastLine = editor.getLine(cursor.line - 1) as string;
      if (lastLine.includes('#')) {
        haveComment = true;
      }
      autoGenCode = await codeGen(lastLine + '\n' + thisLine);
    } else {
      autoGenCode = await codeGen(thisLine);
    }
    console.log('haveComment', haveComment);

    /**
     * Adjustment strategy
     */
    // if (haveComment) {
    //   autoGenCode = await codeGen(lastLine + '\n' + thisLine);
    // } else {
    //   autoGenCode = await codeGen(thisLine);
    // }

    // Create a list of matching tokens. The last one is used for space occupying.
    const tokenList = [
      {
        value: cursorToken.value + autoGenCode,
        offset: cursorToken.offset,
        type: 'codegen'
      },
      {
        value: '',
        offset: cursorToken.offset,
        type: 'codegen'
      }
    ];

    const completionList = tokenList.map(t => t.value);
    // Remove duplicate completions from the list
    const matches = Array.from(new Set<string>(completionList));

    return {
      start: cursorToken.offset,
      end: cursorToken.offset + cursorToken.value.length,
      matches,
      metadata: {}
    };
  }
}
