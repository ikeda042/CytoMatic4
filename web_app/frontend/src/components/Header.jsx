import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';

function Header() {
  return (
    <AppBar position="static" style={{ backgroundColor: 'gray' }}>
      <Toolbar>
        <Typography variant="h6">
          PhenoPixel4.0
        </Typography>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
