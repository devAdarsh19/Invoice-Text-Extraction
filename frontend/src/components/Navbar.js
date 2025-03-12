import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav>
      <div className="nav-container">
        <h1 className="app-name">MyApp</h1>
          <ul className="nav-links">
            <li className="nav-item">About</li>
            <li className="nav-item">Contact</li>
          </ul>
      </div>
    </nav>
  );
};

export default Navbar;
