import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  ContextConnector,
  ICompletionManager,
  KernelConnector
} from '@jupyterlab/completer';

import { ISettingRegistry } from '@jupyterlab/settingregistry';

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';

import { requestAPI } from './handler';

import { CompletionConnector } from './connector';

import { CustomConnector } from './customconnector';

/**
 * The command IDs used by the console plugin.
 */
namespace CommandIDs {
  export const invoke = 'completer:invoke';

  export const invokeNotebook = 'completer:invoke-notebook';

  export const select = 'completer:select';

  export const selectNotebook = 'completer:select-notebook';
}

/**
 * Initialization data for the codegen-paddle extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'codegen-paddle:plugin',
  autoStart: true,
  requires: [ICompletionManager, INotebookTracker, ISettingRegistry],
  activate: async (
    app: JupyterFrontEnd,
    completionManager: ICompletionManager,
    notebooks: INotebookTracker,
    settings: ISettingRegistry
  ) => {
    let max_length = 16;
    let min_length = 0;
    let repetition_penalty = 1.0;
    let top_p = 1.0;
    let top_k = 10;
    let temperature = 0.5;
    let device = 'cpu';
    let model = 'Salesforce/codegen-350M-mono/';
    console.log('JupyterLab extension codegen-paddle is activated!');
    await settings.load('codegen-paddle:completer').then(setting => {
      // Read the settings
      console.log(setting);
      max_length = setting.get('max_length').composite as number;
      min_length = setting.get('min_length').composite as number;
      repetition_penalty = setting.get('repetition_penalty')
        .composite as number;
      top_p = setting.get('top_p').composite as number;
      top_k = setting.get('top_k').composite as number;
      temperature = setting.get('temperature').composite as number;
      device = setting.get('device').composite as string;
      model = setting.get('model').composite as string;
    });

    console.log({
      max_length,
      min_length,
      repetition_penalty,
      top_p,
      top_k,
      temperature,
      device,
      model
    });

    const initParams = {
      max_length,
      min_length,
      repetition_penalty,
      top_p,
      top_k,
      temperature,
      device,
      model
    };
    try {
      const res = await requestAPI<any>('init-model', {
        body: JSON.stringify(initParams),
        method: 'POST'
      });
      console.log(res);
    } catch (reason) {
      console.error(
        `Error on POST /codegen-paddle-backend/init-model ${initParams}.\n${reason}`
      );
    }

    // GET request
    try {
      const data = await requestAPI<any>('hello');
      console.log(data);
    } catch (reason) {
      console.error(`Error on GET /codegen-paddle-backend/hello.\n${reason}`);
    }

    // Modelled after completer-extension's notebooks plugin
    notebooks.widgetAdded.connect(
      (sender: INotebookTracker, panel: NotebookPanel) => {
        let editor = panel.content.activeCell?.editor ?? null;
        const session = panel.sessionContext.session;
        const options = { session, editor };
        const connector = new CompletionConnector([]);
        const handler = completionManager.register({
          connector,
          editor,
          parent: panel
        });

        const updateConnector = () => {
          editor = panel.content.activeCell?.editor ?? null;
          options.session = panel.sessionContext.session;
          options.editor = editor;
          handler.editor = editor;

          const kernel = new KernelConnector(options);
          const context = new ContextConnector(options);
          const custom = new CustomConnector(options);
          handler.connector = new CompletionConnector([
            kernel,
            context,
            custom
          ]);
        };

        // Update the handler whenever the prompt or session changes
        panel.content.activeCellChanged.connect(updateConnector);
        panel.sessionContext.sessionChanged.connect(updateConnector);
      }
    );

    // Add notebook completer command.
    app.commands.addCommand(CommandIDs.invokeNotebook, {
      execute: () => {
        const panel = notebooks.currentWidget;
        if (panel && panel.content.activeCell?.model.type === 'code') {
          return app.commands.execute(CommandIDs.invoke, { id: panel.id });
        }
      }
    });

    // Add notebook completer select command.
    app.commands.addCommand(CommandIDs.selectNotebook, {
      execute: () => {
        const id = notebooks.currentWidget && notebooks.currentWidget.id;

        if (id) {
          return app.commands.execute(CommandIDs.select, { id });
        }
      }
    });

    // Set enter key for notebook completer select command.
    app.commands.addKeyBinding({
      command: CommandIDs.selectNotebook,
      keys: ['Enter'],
      selector: '.jp-Notebook .jp-mod-completer-active'
    });
  }
};

export default plugin;
