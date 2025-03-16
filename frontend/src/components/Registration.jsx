import React, {useState} from 'react'

const Registration = () => {
  return (
    <div>
      <form action="" method="post">
        <input type="email" name="user-email" required/>
        <input type="password" name="password" required />
        <button type="submit">Submit</button>
      </form>
    </div>
  )
}

export default Registration
