# STL Litíase

Esse projeto tem como objetivo utilizar a estratégia STL para identificar litíase renal em imagens de tomografias computadorizada

## Installation

Usando [pip](https://pip.pypa.io/en/stable/) para instalar rode o seguinte comando, utilizando Python 3.11.X:

```bash
pip install -r requirements.txt
```

O pacote matplotlib possui uma dependência do [MSBuild](https://learn.microsoft.com/pt-br/visualstudio/msbuild/walkthrough-using-msbuild?view=vs-2022), o que se faz necessário a instalação dele, e colocar ele na [PATH](https://stackoverflow.com/questions/6319274/how-do-i-run-msbuild-from-the-command-line-using-windows-sdk-7-1) do sistema.


## Uso

Para rodar o código basta rodar o script stl_litiase.py, passando o arquivo de configurações como parâmetro, sem a extensão. Esse arquivo precisa estar dentro do projeto, no diretório 'CONFIGURACOES'.

```bash
python stl_litiase.py padrao criar
```

Se quiser criar um arquivo de configurações novo, basta fazer uma cópia do arquivo 'padrao.json', alterar o nome e seguir a mesma estrutura do JSON.

## Reprodutibilidade
Para ser possível garantir reprodutibilidade com os resultados, basta utilizar o arquivo 'padrao.json' ao rodar o código,
não alterando nenhum parâmetro.

## Código de erros
Ao rodar o código e dar algum erro, irá sair com um código e uma breve descrição. Segue lista de erros e suas explicações

| Código | Descrição                                                                                          | Sugestão                                                                                                                               |
|:------:|:---------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| **-1** | O arquivo de configurações passado como parãmetro não foi encontrado no diretório 'CONFIGURACOES'. | Certifiqui-se que o arquivo está realmente no diretório, se o seu nome está certo, e que você não passou a extensão ao rodar o script. |
| **-2** | A variável que é usada no código não foi encontrada no arquivo de configurações.                   | Certifique-se que a variável não foi deletada ou teve seu nome editado por engano no arquivo de configurações.                         |
| **-3** | O tipo de processo não é valido                                                                    | Certifique-se que o segundo parâmetro é 'criar' ou 'classificar'                                                                       |
| **-4** |                                                                                                    |                                                                                                                                        |
| **-5** |                                                                                                    |                                                                                                                                        |
| **-6** |                                                                                                    |                                                                                                                                        |
| **-7** |                                                                                                    |                                                                                                                                        |


## License

[MIT](https://choosealicense.com/licenses/mit/)