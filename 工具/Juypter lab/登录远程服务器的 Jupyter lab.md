# 登录远程服务器

登录远程服务器并进入指定路径（作为 jupyter lab 的根目录）

# 配置服务器的 jupyter lab

1.  【服务器】输入 `jupyter notebook password`设置 jupyter lab 登录密码，这里输入密码时不可见。记住这个密码，以后在本地登录远程 jupyter lab 时需要使用。

    ![image-20201006142705506](https://gitee.com/llillz/images/raw/master/image-20201006142705506.png)

    此外，在服务器还会生成一个 `jupyter_notebook_config.json` 配置文件，记住这个路径，后面还会使用。

2.  【服务器】在 `~/.jupyter/jupyter_notebook_config.json` 中写入以下内容

    ```json
    {
      "NotebookApp": {
        "ip": "*",
        "open_browser": false
      }
    }
    ```

    `"ip": "*",` 是让服务器监听所有 ip。

3.  【服务器】**输入 `jupyter lab --port LAB_PORT --no-browser`启动 jupyter lab**。**注意**。这里的 `LAB_PORT` 需要自己指定，一般可以是 `8889`。

4.  【本地】打开浏览器，输入 `remote__ip:port`就可以访问到服务器的 jupyter lab 了。注意，`remote_ip` 是你服务器的 ip，`port` 就是第 3 步设置的 `LAB_PORT`。如果一切正常，此时浏览器会提示你输入密码：

    ![image-20201006143521153](https://gitee.com/llillz/images/raw/master/image-20201006143521153.png)

    输入第 1 步设置的密码，就能使用 jupyter lab 了！

![image-20201006143615107](https://gitee.com/llillz/images/raw/master/image-20201006143615107.png)



# Reference

1.  https://alanlee.fun/2018/07/14/remote-jupyter/
2.  https://jupyter-notebook.readthedocs.io/en/latest/public_server.html