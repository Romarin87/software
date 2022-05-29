#!/bin/bash
wget -c https://github.com/tmux/tmux/releases/download/3.2/tmux-3.2.tar.gz
wget -c https://github.com/libevent/libevent/releases/download/release-2.1.11-stable/libevent-2.1.11-stable.tar.gz
wget -c https://ftp.gnu.org/gnu/ncurses/ncurses-6.2.tar.gz --no-check-certificate
tar -xzvf tmux-3.2.tar.gz
tar -xzvf libevent-2.1.11-stable.tar.gz
tar -xzvf ncurses-6.2.tar.gz
cd libevent-2.1.11-stable
./configure --prefix=$HOME/tmux_depend --disable-shared
make && make install
cd  ../ncurses-6.2
./configure --prefix=$HOME/tmux_depend
make && make install
cd  ../tmux-3.2
./configure CFLAGS="-I$HOME/tmux_depend/include -I/$HOME/tmux_depend/include/ncurses" LDFLAGS="-L/$HOME/tmux_depend/lib -L/$HOME/tmux_depend/include/ncurses -L/$HOME/tmux_depend/include"
make
cp tmux  $HOME/tmux_depend/bin
echo export PATH=$HOME/tmux_depend/bin:'$PATH' >> $HOME/.bashrc
source $HOME/.bashrc